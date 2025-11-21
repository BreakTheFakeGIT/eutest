import os
import re
import asyncio
import numpy as np
import time
import functools
import psycopg
from psycopg import sql
from typing import List, Callable, Optional, List, Union
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from src.utils.text_splitter import process_cut_text
from src.utils.select import db_select_interptacje
from src.prompts.prompt_taxes_v2a import TAX_PROMPTS


import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


###################################
TBL_INTERP = os.environ.get("TBL_INTERP")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################

# --- Konfiguracja logowania ---
logger = logger_utils.setup_logger(name=f'eu_llm')
query_sql = sql.SQL("""SELECT
                    tresc_interesariusz,
                    id AS interpretacje_id,
                    id_informacji,
                    typ_podatku
                FROM public.interpretacje AS ta
                WHERE 1=1
                    AND kategoria_informacji = 1
                    AND szablonid IN (1,2)
                    AND EXISTS(SELECT 1 FROM public.interpretacje_podobne AS tb WHERE ta.id_informacji = tb.id_informacji_powiazane)
                ORDER BY id_informacji DESC
                LIMIT 100""")

def print_chunk(chunk: str):
    print(chunk, end="", flush=True)

def log_chunk(chunk: str):
    with open("stream_log.txt", "a", encoding="utf-8") as f:
        f.write(chunk)



class LLMWrapper:
    def __init__(self, model_name="hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0",
                 temperature=0.7,
                 max_tokens=1024,
                 max_retries=3,
                 retry_delay=2
                 ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens)

    def _timed(self, func):
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Async method '%s' executed in %.4f seconds", func.__name__, elapsed)
            return result

        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info("Sync method '%s' executed in %.4f seconds", func.__name__, elapsed)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


    # Retry Helper
    def _retry(self, func, *args, **kwargs):
        """Generic retry logic for sync operations."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[Attempt {attempt}] Error: {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))  # exponential backoff
                    logger.error(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    async def _retry_async(self, func, *args, **kwargs):
        """Generic retry logic for async operations."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[Attempt {attempt}] Async Error: {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))  # exponential backoff
                    logger.error(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

    # Synchronous Methods
    def invoke(self, prompt: str):
        """Synchronous full response with retries."""
        return self._retry(self.llm.invoke, prompt)

    def stream_invoke(self, prompt: str, callbacks: List[Callable[[str], None]]):
        """Synchronous streaming with retries and multiple callbacks."""
        def stream_func():
            for chunk in self.llm.stream(prompt):
                for callback in callbacks:
                    callback(chunk)
        return self._retry(stream_func)

    # Asynchronous Methods
    async def stream_invoke_async(self, prompt: str, callbacks: List[Callable[[str], None]]):
        """Async streaming with retries and multiple callbacks."""
        async def stream_func():
            async for chunk in self.llm.astream(prompt):
                await asyncio.gather(*[
                    asyncio.get_event_loop().run_in_executor(None, cb, chunk)
                    for cb in callbacks
                ])
        return await self._retry_async(stream_func)

    # Model Management
    def change_model(self, new_model: str):
        """Change the model dynamically."""
        self.model_name = new_model
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens)
        print(f"Model changed to: {self.model_name}")


def stream_texts_to_llm(conn_str,
                        llm_wrapper,
                        #cutter,
                        query="SELECT text FROM documents",
                        batch_size=10,
                        max_retries=3
                        ):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor(name="stream_cursor", row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query)
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                texts = [row.get('tresc_interesariusz', []) for row in rows]
                id_infos = [row.get('id_informacji', []) for row in rows]
                tax_types = [row.get('typ_podatku', []) for row in rows]

                for id_info, tax_type, text in zip(id_infos,tax_types,texts):
                    for attempt in range(1, max_retries + 1):
                        try:
                            text = process_cut_text(text)
                            prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
                            text = prompt_template.format(user_text=text)
                            llm_wrapper.stream_invoke(prompt=text, callbacks=[lambda chunk: print(chunk, end="")])

                            break
                        except Exception as e:
                            logger.error(f"Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
                            if attempt < max_retries:
                                delay = 2 ** attempt
                                logger.warning(f"Retrying in {delay}s...")
                                time.sleep(delay)
                            else:
                                logger.error(f"Skipping text after {max_retries} attempts: {text[:50]}")

async def async_stream_texts_to_llm_async(conn_str,
                                        llm_wrapper,
                                        query="SELECT text FROM documents",
                                        batch_size=100,
                                        max_retries=3):
    async with await psycopg.AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor(name="stream_cursor",row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(query)
            while True:
                rows = await cur.fetchmany(batch_size)
                if not rows:
                    break
                texts = [row.get('tresc_interesariusz', []) for row in rows]
                id_infos = [row.get('id_informacji', []) for row in rows]
                tax_types = [row.get('typ_podatku', []) for row in rows]

                for id_info, tax_type, text in zip(id_infos,tax_types,texts):
                    for attempt in range(1, max_retries + 1):
                        try:
                            await text = process_cut_text(text)
                            prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
                            text = prompt_template.format(user_text=text)

                            print(id_info)
                            print(tax_type)
                            await llm_wrapper.stream_invoke_async(prompt=text, callbacks=[lambda chunk: print(chunk, end="")])



                            break
                        except Exception as e:
                            logger.error(f"Async Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
                            if attempt < max_retries:
                                delay = 2 ** attempt
                                logger.warning(f"Retrying in {delay}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Skipping text after {max_retries} attempts: {text[:50]}")


async def main():
    llm_bielik = LLMWrapper(max_retries=3, retry_delay=2)
    await async_stream_texts_to_llm_async(conn_str=POSTGRES_DSN,
                                          llm_wrapper=llm_bielik,
                                          query=query_sql,
                                          batch_size=10,
                                          max_retries=3)
    # formatted_prompt = test_eu_llm()
    #text_cutter = TextCutter()
    # stream_texts_to_llm(conn_str=POSTGRES_DSN,
    #                     llm_wrapper=llm_bielik,
    #                     query=query_sql,
    #                     batch_size=10,
    #                     max_retries=3
    #                     )

    # # Sync full response with retries
    # print("Full response:")
    # print(llm_wrapper.invoke(prompt=formatted_prompt))

    # # Sync streaming with retries
    # print("\nSync streaming response:")
    # llm_wrapper.stream_invoke(prompt=formatted_prompt, callbacks=[print_chunk, log_chunk])

    # # Async streaming with retries
    # print("\nAsync streaming response:")
    # await llm_wrapper.stream_invoke_async(prompt=formatted_prompt, callbacks=[print_chunk])

    # # Change model dynamically
    # llm_wrapper.change_model(new_model="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M")
    # print("\nAsync streaming response from llama2:")
    # await llm_wrapper.stream_invoke_async(prompt=formatted_prompt, callbacks=[print_chunk])

if __name__ == "__main__":
    asyncio.run(main())








# # --- 1. Przypisanie promptu (w języku polskim) ---
# # def process_tax_text(
# #     user_text: str,
# #     tax_type: str,
    
# #     # conn: psycopg.Connection,
# #     # llm: OllamaLLM,
# #     # embedding_model: SentenceTransformer
# # ) -> Optional[Dict[str, Any]]:
# #     """"""
# #     logger.info(f"Tax type: {tax_type}")

# #     # --- 1. Przypisanie promptu ---
# #     prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
# #     formatted_prompt = prompt_template.format(user_text=user_text)
# #     logger.info(f"Formatted prompt: {formatted_prompt}")

# #     llm = LLMWrapper(model_name="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M", temperature=0.7, max_tokens=1024)
# #     llm.invoke(prompt=formatted_prompt)


# #     #logger.info(f"Zakończono przetwarzanie dla ID: {new_id}.")
# #     #return results_dict


# # # --- Przykładowe użycie ---
# # def main():
# #     user_text = "Główna funkcja do uruchomienia przykładu."
# #     tax_type = "cit"
# #     process_tax_text(user_text=user_text,tax_type=tax_type)
# #     # # --- Inicjalizacja modeli ---
#     # try:
#     #     logger.info("Inicjalizacja modelu LLM (Ollama)...")

#     #     logger.info("Inicjalizacja modelu SentenceTransformer...")
#     #     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     #     # ZMIANA: Pobierz wymiar wektora z modelu
#     #     dimension = embedding_model.get_sentence_embedding_dimension()
#     #     logger.info(f"Model SentenceTransformer gotowy. Wymiar wektora: {dimension}")

#     # except Exception as e:
#     #     logger.error(f"Błąd podczas inicjalizacji modeli: {e}")
#     #     logger.error("Upewnij się, że Ollama jest uruchomiona i masz model 'bielik'.")
#     #     logger.error("Upewnij się, że biblioteka 'sentence-transformers' jest zainstalowana.")
#     #     return

#     # # --- Połączenie z bazą danych (Psycopg 3) ---
#     # DB_CONN_STRING = "dbname=postgres user=postgres password=mysecretpassword host=localhost port=5432"


# if __name__ == "__main__":


#     rows = db_select_interptacje(conn_str=POSTGRES_DSN, table_name=TBL_INTERP)
#     for row in rows[0:1]:
#         # User choice
#         id_informacji = row.get('id_informacji', [])
#         tax_type = row.get('typ_podatku', [])
#         teza_text = row.get('teza', [])
#         interp_text = row.get('tresc_interesariusz', [])
#         interp_text = cut_text_start_by_regex(interp_text)
#         interp_text = cut_text_by_regex(interp_text)
#         logger.info(f"Tax type: {tax_type}")

#     # --- 1. Przypisanie promptu ---
#         prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
#         formatted_prompt = prompt_template.format(user_text=interp_text)
#         print(f"Formatted prompt: {formatted_prompt}")

#         #"hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0"
#         #"hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M"
#         llm = LLMWrapper(model_name="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M",temperature=0.7, max_tokens=1024)
#         llm.invoke(prompt=formatted_prompt)



    #     template = """Jestem ekspertem skarbowym. Otrzymałem treść wniosku: {question}

    #     Odpowiedz na pytania:
    #     1) jakie są słowa kluczowe
    #     2) stwórz 5 pytań do wypowiedzi
    #     Wyniki zwróć w postaci listy."""

#"hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M"
    #     prompt = ChatPromptTemplate.from_template(template)

    #     model = OllamaLLM(model="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M", temperature=0.9, max_tokens=1024)
    #     chain = prompt | model

    #     response = chain.invoke({"question": interp_text})
    #     print(response)

    #main()


# async def main():
#     llm_wrapper = BielikLLMWrapper()
#     print("Full response:")
#     print(llm_wrapper.invoke("Hello from Bielik!"))

#     print("\nAsync streaming response:")
#     await llm_wrapper.stream_invoke_async(
#         "Write a short poem about AI.",
#         callbacks=[print_chunk, log_chunk]
#     )

#     llm_wrapper.change_model("llama2")
#     print("\nStreaming response from llama2:")
#     await llm_wrapper.stream_invoke_async(
#         "Explain quantum computing in simple terms.",
#         callbacks=[print_chunk]
#     )

# if __name__ == "__main__":
#     asyncio.run(main())










# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama.llms import OllamaLLM


# def test_plumm1():

#     template = """Question: {question}

#     Answer: Pomyslę i odpowiem po polsku."""

#     prompt = ChatPromptTemplate.from_template(template)

#     model = OllamaLLM(model="PRIHLOP/PLLuM:12b", temperature=0.9, max_tokens=1024)

#     chain = prompt | model

#     response = chain.invoke({"question": "Jakie są rodzaje podatków w Polsce?"})
#     print(response)





# if __name__ == "__main__":


#     rows = db_select_interptacje(conn_str=POSTGRES_DSN, table_name=TBL_INTERP)
#     for row in rows[0:1]:
#         # User choice
#         id_informacji = row.get('id_informacji', [])
#         tax_type = row.get('typ_podatku', [])
#         teza_text = row.get('teza', [])
#         interp_text = row.get('tresc_interesariusz', [])

#         interp_text = cut_text_start_by_regex(interp_text)
#         interp_text = cut_text_by_regex(interp_text)

#         template = """Jestem ekspertem skarbowym. Otrzymałem treść wniosku: {question}

#         Odpowiedz na pytania:
#         1) jakie są słowa kluczowe
#         2) stwórz 5 pytań do wypowiedzi
#         Wyniki zwróć w postaci listy."""
#         prompt = ChatPromptTemplate.from_template(template)

#         model = OllamaLLM(model="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M", temperature=0.9, max_tokens=1024)
#         chain = prompt | model

#         response = chain.invoke({"question": interp_text})
#         print(response)

#         # interp_text = split_text_by_header_regex(text=interp_text)

# #hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0
#         # print("\n" + "=" * 50)
#         # print("### EXPERIMENT  ###")
#         # print("=" * 50)
#         # results = experiment_text_splitting(
#         #     text=interp_text,
#         #     chunk_sizes=[1000],
#         #     chunk_overlaps=[200],
#         # )
#         # print(results)

# async def stream_texts_to_llm_async(conn_str,
#                                     llm_wrapper,
#                                     query=query_sql,
#                                     batch_size=100,
#                                     max_retries=3
#                                     ):
#     async with await psycopg.AsyncConnection.connect(conn_str) as conn:
#         async with conn.cursor(name="stream_cursor", row_factory=psycopg.rows.dict_row) as cur:
#             await cur.execute(query)
#             while True:
#                 rows = await cur.fetchmany(batch_size)
#                 if not rows:
#                     break
#                 texts = [row[0] for row in rows]

#                 for text in texts:
#                     for attempt in range(1, max_retries + 1):
#                         try:
#                             await llm_wrapper.stream_invoke_async(text, callbacks=[lambda chunk: print(chunk, end="")])
#                             break
#                         except Exception as e:
#                             logger.error(f"Async Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
#                             if attempt < max_retries:
#                                 delay = 2 ** attempt
#                                 logger.warning(f"Retrying in {delay}s...")
#                                 await asyncio.sleep(delay)
#                             else:
#                                 logger.error(f"Skipping text after {max_retries} attempts: {text[:50]}")




# def test_eu_llm(): 
#     rows = db_select_interptacje(conn_str=POSTGRES_DSN, table_name=TBL_INTERP)
#     for row in rows[0:1]:
#         # User choice
#         id_informacji = row.get('id_informacji', [])
#         tax_type = row.get('typ_podatku', [])
#         teza_text = row.get('teza', [])
#         interp_text = row.get('tresc_interesariusz', [])
#         interp_text = cut_text_start_by_regex(interp_text)
#         interp_text = cut_text_by_regex(interp_text)
#         logger.info(f"Tax type: {tax_type}")
#         # --- 1. Przypisanie promptu ---
#         prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
#         formatted_prompt = prompt_template.format(user_text=interp_text)
#         print(f"Formatted prompt: {formatted_prompt}")
#     return formatted_prompt