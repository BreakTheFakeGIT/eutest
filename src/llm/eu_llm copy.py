import os
import asyncio
import numpy as np
import psycopg
from typing import List, Callable, Optional
#from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import math
#from sentence_transformers import SentenceTransformer
from langchain_ollama.llms import OllamaLLM
from src.utils.text_splitter import cut_text_start_by_regex, cut_text_by_regex
from src.utils.select import db_select_interptacje
from src.prompts.prompt_taxes_v2 import TAX_PROMPTS

import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


###################################
TBL_INTERP = os.environ.get("TBL_INTERP")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################

# --- Konfiguracja logowania ---
logger = logger_utils.setup_logger(name=f'eu_llm')

def print_chunk(chunk: str):
    print(chunk, end="", flush=True)

def log_chunk(chunk: str):
    with open("stream_log.txt", "a", encoding="utf-8") as f:
        f.write(chunk)


class LLMWrapper:
    def __init__(self, model_name="hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0", temperature=0.7, max_tokens=1024,max_retries=3, retry_delay=2):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = OllamaLLM(model=self.model_name, temperature=self.temperature, max_tokens=self.max_tokens)

    # Retry Helper
    def _retry(self, func, *args, **kwargs):
        """Generic retry logic for sync operations."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[Attempt {attempt}] Error: {e}")
                if attempt < self.max_retries:
                    print(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    raise

    async def _retry_async(self, func, *args, **kwargs):
        """Generic retry logic for async operations."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"[Attempt {attempt}] Async Error: {e}")
                if attempt < self.max_retries:
                    print(f"Retrying in {self.retry_delay}s...")
                    await asyncio.sleep(self.retry_delay)
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




class HybridBatchProcessor:
    def __init__(self, llm_wrapper, embed_fn: Optional[Callable[[str], List[float]]] = None,
                 conn_str: Optional[str] = None, max_retries: int = 3, base_delay: int = 2):
        """
        :param llm_wrapper: Instance of BielikLLMWrapper.
        :param embed_fn: Function to generate embeddings for text.
        :param conn_str: PostgreSQL connection string for pgvector.
        :param max_retries: Max retry attempts for DB operations.
        :param base_delay: Base delay for exponential backoff.
        """
        self.llm_wrapper = llm_wrapper
        self.embed_fn = embed_fn
        self.conn_str = conn_str
        self.max_retries = max_retries
        self.base_delay = base_delay

    # Retry Helpers
    def _retry(self, func, *args, **kwargs):
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"[DB Attempt {attempt}] Error: {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))  # exponential backoff
                    print(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise

    async def _retry_async(self, func, *args, **kwargs):
        for attempt in range(1, self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"[DB Attempt {attempt}] Async Error: {e}")
                if attempt < self.max_retries:
                    delay = self.base_delay * (2 ** (attempt - 1))  # exponential backoff
                    print(f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

    # Vectorization
    def _vectorize(self, text: str) -> Optional[List[float]]:
        return self.embed_fn(text) if self.embed_fn else None

    # Sync Batch Processing
    def process_sync(self, texts: List[str], batch_size: int = 50, preprocess_fn: Optional[Callable[[str], str]] = None,
                     callbacks: Optional[List[Callable[[str], None]]] = None, max_workers: int = 5, vectorize: bool = False,
                     save_to_db: bool = False):
        results, vectors = [], []

        def process_item(text):
            if preprocess_fn:
                text = preprocess_fn(text)
            response = self.llm_wrapper.invoke(text)
            if callbacks:
                for cb in callbacks:
                    cb(response)
            vec = self._vectorize(text) if vectorize else None
            return text, response, vec

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, item) for batch in batches for item in batch]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Sync"):
                text, res, vec = future.result()
                results.append(res)
                if vec is not None:
                    vectors.append(vec)

        if save_to_db and self.conn_str and vectors:
            self._retry(self._save_embeddings_pgvector, texts, np.array(vectors))

        return results, np.array(vectors) if vectorize else None

    # -------------------------
    # Async Batch Processing
    # -------------------------
    async def process_async(self, texts: List[str], batch_size: int = 50, preprocess_fn: Optional[Callable[[str], str]] = None,
                             callbacks: Optional[List[Callable[[str], None]]] = None, concurrency_limit: int = 10,
                             vectorize: bool = False, save_to_db: bool = False):
        results, vectors = [], []
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_item(text):
            async with semaphore:
                if preprocess_fn:
                    text = preprocess_fn(text)
                response = await self.llm_wrapper.stream_invoke_async(text, callbacks or [])
                vec = self._vectorize(text) if vectorize else None
                return text, response, vec

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        for batch in tqdm(batches, desc="Processing Async Batches"):
            tasks = [asyncio.create_task(process_item(item)) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            for text, res, vec in batch_results:
                results.append(res)
                if vec is not None:
                    vectors.append(vec)

        if save_to_db and self.conn_str and vectors:
            await self._retry_async(self._save_embeddings_pgvector_async, texts, np.array(vectors))

        return results, np.array(vectors) if vectorize else None

    # -------------------------
    # Save Embeddings to Postgres
    # -------------------------
    def _save_embeddings_pgvector(self, texts: List[str], embeddings: np.ndarray):
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                for text, emb in tqdm(zip(texts, embeddings), total=len(texts), desc="Saving to DB"):
                    emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                    cur.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)", (text, emb_str))
            conn.commit()

    async def _save_embeddings_pgvector_async(self, texts: List[str], embeddings: np.ndarray):
        async with await psycopg.AsyncConnection.connect(self.conn_str) as conn:
            async with conn.cursor() as cur:
                for text, emb in tqdm(zip(texts, embeddings), total=len(texts), desc="Saving to DB (Async)"):
                    emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                    await cur.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)", (text, emb_str))
            await conn.commit()

    # -------------------------
    # Similarity Search
    # -------------------------
    def similarity_search(self, query_embedding: List[float], top_k: int = 5):
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        return self._retry(self._similarity_query, emb_str, top_k)

    async def similarity_search_async(self, query_embedding: List[float], top_k: int = 5):
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        return await self._retry_async(self._similarity_query_async, emb_str, top_k)

    def _similarity_query(self, emb_str: str, top_k: int):
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text, embedding <-> %s::vector AS distance
                    FROM documents
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """, (emb_str, emb_str, top_k))
                return cur.fetchall()

    async def _similarity_query_async(self, emb_str: str, top_k: int):
        async with await psycopg.AsyncConnection.connect(self.conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT id, text, embedding <-> %s::vector AS distance
                    FROM documents
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """, (emb_str, emb_str, top_k))
                return await cur.fetchall()
















async def async_stream_texts_to_llm_async(conn_str, llm_wrapper, query="SELECT text FROM documents", batch_size=100, max_retries=3):
    async with await psycopg.AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor(name="stream_cursor") as cur:
            await cur.execute(query)
            while True:
                rows = await cur.fetchmany(batch_size)
                if not rows:
                    break
                texts = [row[0] for row in rows]

                for text in texts:
                    for attempt in range(1, max_retries + 1):
                        try:
                            await llm_wrapper.stream_invoke_async(text, callbacks=[lambda chunk: print(chunk, end="")])
                            break
                        except Exception as e:
                            logger.error(f"Async Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
                            if attempt < max_retries:
                                delay = 2 ** attempt
                                logger.warning(f"Retrying in {delay}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Skipping text after {max_retries} attempts: {text[:50]}")



def stream_texts_to_llm(conn_str, llm_wrapper, query="SELECT text FROM documents", batch_size=100, max_retries=3):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor(name="stream_cursor") as cur:
            cur.execute(query)
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break
                texts = [row[0] for row in rows]

                for text in texts:
                    for attempt in range(1, max_retries + 1):
                        try:
                            llm_wrapper.stream_invoke(text, callbacks=[lambda chunk: print(chunk, end="")])
                            break
                        except Exception as e:
                            logger.error(f"Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
                            if attempt < max_retries:
                                delay = 2 ** attempt
                                logger.warning(f"Retrying in {delay}s...")
                                time.sleep(delay)
                            else:
                                logger.error(f"Skipping text after {max_retries} attempts: {text[:50]}")
class Timing:
    def __init__(self, log_to_file=None, warning_log=None, threshold=None):
        """
        log_to_file: Optional path to a file for all timings.
        warning_log: Optional path to a file for warnings only.
        threshold: Optional time in seconds; if exceeded, a warning is logged.
        """
        self.log_to_file = log_to_file
        self.warning_log = warning_log
        self.threshold = threshold

        # ANSI color codes
        self.COLOR_RESET = "\033[0m"
        self.COLOR_WARNING = "\033[91m"  # Red for warnings
        self.COLOR_INFO = "\033[94m"     # Blue for normal info

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                self._report(func.__name__, elapsed)
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                self._report(func.__name__, elapsed)
                return result
            return sync_wrapper

    def _report(self, name, elapsed):
        message = f"Function '{name}' executed in {elapsed:.4f} seconds"
        print(f"{self.COLOR_INFO}{message}{self.COLOR_RESET}")

        # Log all timings if file provided
        if self.log_to_file:
            with open(self.log_to_file, "a") as f:
                f.write(message + "\n")

        # Handle threshold warnings
        if self.threshold and elapsed > self.threshold:
            warning_msg = f"{message} ⚠️ WARNING: Exceeded threshold of {self.threshold:.4f} seconds"
            print(f"{self.COLOR_WARNING}{warning_msg}{self.COLOR_RESET}")
            if self.warning_log:
                with open(self.warning_log, "a") as f:
                    f.write(warning_msg + "\n")























def test_eu_llm(): 
    rows = db_select_interptacje(conn_str=POSTGRES_DSN, table_name=TBL_INTERP)
    for row in rows[0:1]:
        # User choice
        id_informacji = row.get('id_informacji', [])
        tax_type = row.get('typ_podatku', [])
        teza_text = row.get('teza', [])
        interp_text = row.get('tresc_interesariusz', [])
        interp_text = cut_text_start_by_regex(interp_text)
        interp_text = cut_text_by_regex(interp_text)
        logger.info(f"Tax type: {tax_type}")
        # --- 1. Przypisanie promptu ---
        prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
        formatted_prompt = prompt_template.format(user_text=interp_text)
        print(f"Formatted prompt: {formatted_prompt}")
    return formatted_prompt





async def main():

    llm_wrapper = LLMWrapper(max_retries=3, retry_delay=2)
    formatted_prompt = test_eu_llm()

    # Sync full response with retries
    print("Full response:")
    print(llm_wrapper.invoke(prompt=formatted_prompt))

    # Sync streaming with retries
    print("\nSync streaming response:")
    llm_wrapper.stream_invoke(prompt=formatted_prompt, callbacks=[print_chunk, log_chunk])

    # Async streaming with retries
    print("\nAsync streaming response:")
    await llm_wrapper.stream_invoke_async(prompt=formatted_prompt, callbacks=[print_chunk])

    # Change model dynamically
    llm_wrapper.change_model(new_model="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M")
    print("\nAsync streaming response from llama2:")
    await llm_wrapper.stream_invoke_async(prompt=formatted_prompt, callbacks=[print_chunk])

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

class HybridBatchProcessor:
    def __init__(self, llm_wrapper, embed_fn: Optional[Callable[[str], List[float]]] = None,
                 conn_str: Optional[str] = None, max_retries: int = 3, base_delay: int = 2):
        """
        :param llm_wrapper: Instance of LLMWrapper.
        :param embed_fn: Function to generate embeddings for text.
        :param conn_str: PostgreSQL connection string for pgvector.
        :param max_retries: Max retry attempts for DB operations.
        :param base_delay: Base delay for exponential backoff.
        """
        self.llm_wrapper = llm_wrapper
        self.embed_fn = embed_fn
        self.conn_str = conn_str
        self.max_retries = max_retries
        self.base_delay = base_delay


    # Vectorization
    def _vectorize(self, text: str) -> Optional[List[float]]:
        return self.embed_fn(text) if self.embed_fn else None

    # Sync Batch Processing
    def process_sync(self, texts: List[str], batch_size: int = 50, preprocess_fn: Optional[Callable[[str], str]] = None,
                     callbacks: Optional[List[Callable[[str], None]]] = None, max_workers: int = 5, vectorize: bool = False,
                     save_to_db: bool = False):
        results, vectors = [], []

        def process_item(text):
            if preprocess_fn:
                text = preprocess_fn(text)
            response = self.llm_wrapper.invoke(text)
            if callbacks:
                for cb in callbacks:
                    cb(response)
            vec = self._vectorize(text) if vectorize else None
            return text, response, vec

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        # Flatten the batches into a single list
        items = [item for batch in batches for item in batch]
        # Adjust workers if needed
        workers = max_workers if max_workers else cpu_count()

        with Pool(processes=workers) as pool:
            # Map the function to all items
            for text, res, vec in tqdm(pool.imap(process_item, items), total=len(items), desc="Processing Sync"):
                results.append(res)
                if vec is not None:
                    vectors.append(vec)



        if save_to_db and self.conn_str and vectors:
            self._retry(self._save_embeddings_pgvector, texts, np.array(vectors))

        return results, np.array(vectors) if vectorize else None

    # Async Batch Processing
    async def process_async(self, texts: List[str], batch_size: int = 50, preprocess_fn: Optional[Callable[[str], str]] = None,
                             callbacks: Optional[List[Callable[[str], None]]] = None, concurrency_limit: int = 10,
                             vectorize: bool = False, save_to_db: bool = False):
        results, vectors = [], []
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def process_item(text):
            async with semaphore:
                if preprocess_fn:
                    text = preprocess_fn(text)
                response = await self.llm_wrapper.stream_invoke_async(text, callbacks or [])
                vec = self._vectorize(text) if vectorize else None
                return text, response, vec

        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        for batch in tqdm(batches, desc="Processing Async Batches"):
            tasks = [asyncio.create_task(process_item(item)) for item in batch]
            batch_results = await asyncio.gather(*tasks)
            for text, res, vec in batch_results:
                results.append(res)
                if vec is not None:
                    vectors.append(vec)

        if save_to_db and self.conn_str and vectors:
            await self._retry_async(self._save_embeddings_pgvector_async, texts, np.array(vectors))

        return results, np.array(vectors) if vectorize else None

    # Save Embeddings to Postgres
    def _save_embeddings_pgvector(self, texts: List[str], embeddings: np.ndarray):
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                for text, emb in tqdm(zip(texts, embeddings), total=len(texts), desc="Saving to DB"):
                    emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                    cur.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)", (text, emb_str))
            conn.commit()

    async def _save_embeddings_pgvector_async(self, texts: List[str], embeddings: np.ndarray):
        async with await psycopg.AsyncConnection.connect(self.conn_str) as conn:
            async with conn.cursor() as cur:
                for text, emb in tqdm(zip(texts, embeddings), total=len(texts), desc="Saving to DB (Async)"):
                    emb_str = "[" + ",".join(str(x) for x in emb) + "]"
                    await cur.execute("INSERT INTO documents (text, embedding) VALUES (%s, %s::vector)", (text, emb_str))
            await conn.commit()

    # Similarity Search
    def similarity_search(self, query_embedding: List[float], top_k: int = 5):
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        return self._retry(self._similarity_query, emb_str, top_k)

    async def similarity_search_async(self, query_embedding: List[float], top_k: int = 5):
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        return await self._retry_async(self._similarity_query_async, emb_str, top_k)

    def _similarity_query(self, emb_str: str, top_k: int):
        with psycopg.connect(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, text, embedding <-> %s::vector AS distance
                    FROM documents
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """, (emb_str, emb_str, top_k))
                return cur.fetchall()

    async def _similarity_query_async(self, emb_str: str, top_k: int):
        async with await psycopg.AsyncConnection.connect(self.conn_str) as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT id, text, embedding <-> %s::vector AS distance
                    FROM documents
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """, (emb_str, emb_str, top_k))
                return await cur.fetchall()


async def stream_texts_to_llm_async(conn_str, llm_wrapper, query="SELECT text FROM documents", batch_size=100, max_retries=3):
    async with await psycopg.AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor(name="stream_cursor") as cur:
            await cur.execute(query)
            while True:
                rows = await cur.fetchmany(batch_size)
                if not rows:
                    break
                texts = [row[0] for row in rows]

                for text in texts:
                    for attempt in range(1, max_retries + 1):
                        try:
                            await llm_wrapper.stream_invoke_async(text, callbacks=[lambda chunk: print(chunk, end="")])
                            break
                        except Exception as e:
                            logger.error(f"Async Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
                            if attempt < max_retries:
                                delay = 2 ** attempt
                                logger.warning(f"Retrying in {delay}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Skipping text after {max_retries} attempts: {text[:50]}")

import re
import asyncio
import logging
import time
from typing import List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class TextCutter:
    def __init__(self, cut_stems: List[str] = None, start_stems: List[str] = None):
        self.cut_stems = cut_stems or [
            'poucze', 'informacja zakres', 'dodatkowe inform', 'informacja o zakres',
            'uzasadnienie interpr', 'postępowanie przed sądami administ',
            'zażalenie na postan', 'podstawa prawna', 'stronie przysługuje prawo',
            'ocena stanowi'
        ]

        self.start_stems = start_stems or [
            "opis stanu faktyczn", "opisu stanu faktyczn", "opisowi stanu faktyczn",
            "opisem stanu faktyczn", "opiśie stanu faktyczn",
            "opis zdarzenia przysz", "opisu zdarzenia przysz", "opisowi zdarzenia przysz",
            "opisem zdarzenia przysz", "opiśie zdarzenia przysz", "pytan"
        ]

        # ✅ Precompile regex patterns for speed
        self.cut_patterns = [re.compile(r'\b' + stem.replace(' ', r'\s+') + r'\w*', re.IGNORECASE)
                             for stem in self.cut_stems]
        self.start_patterns = [re.compile(r'\b' + stem.replace(' ', r'\s+') + r'\w*', re.IGNORECASE)
                                for stem in self.start_stems]

        logging.info("TextCutter initialized with %d cut stems and %d start stems",
                     len(self.cut_stems), len(self.start_stems))

    # -------------------------
    # Utility: Timing decorator
    # -------------------------
    def _timed(self, func):
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logging.info("Async method '%s' executed in %.4f seconds", func.__name__, elapsed)
            return result

        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logging.info("Sync method '%s' executed in %.4f seconds", func.__name__, elapsed)
            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    # -------------------------
    # SYNC METHODS
    # -------------------------
    @_timed
    def cut_text_by_regex(self, text: str) -> str:
        for pattern in self.cut_patterns:
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
        return text

    @_timed
    def cut_text_start_by_regex(self, text: str) -> str:
        indices = [pattern.search(text) for pattern in self.start_patterns]
        positions = [m.start() for m in indices if m]
        start_index = min(positions) if positions else 0
        return text[start_index:]

    # -------------------------
    # ASYNC METHODS
    # -------------------------
    @_timed
    async def cut_text_by_regex_async(self, text: str) -> str:
        for pattern in self.cut_patterns:
            match = pattern.search(text)
            if match:
                text = text[:match.start()]
        return text

    @_timed
    async def cut_text_start_by_regex_async(self, text: str) -> str:
        tasks = [self._search_async(text, pattern) for pattern in self.start_patterns]
        results = await asyncio.gather(*tasks)
        positions = [idx for idx in results if idx > 0]
        start_index = min(positions) if positions else 0
        return text[start_index:]

    async def _search_async(self, text: str, pattern: re.Pattern) -> int:
        match = pattern.search(text)
        return match.start() if match else 0

    @_timed
    async def process_batch_async(self, texts: List[str], mode: str = "start") -> List[str]:
        if mode == "start":
            tasks = [self.cut_text_start_by_regex_async(text) for text in texts]
        elif mode == "cut":
            tasks = [self.cut_text_by_regex_async(text) for text in texts]
        else:
            raise ValueError("Invalid mode. Use 'start' or 'cut'.")
        return await asyncio.gather(*tasks)

    # -------------------------
    # HYBRID METHOD
    # -------------------------
    @_timed
    def process_batch(self, texts: List[str], mode: str = "start", threshold: int = 10) -> Union[List[str], List[str]]:
        if len(texts) <= threshold:
            logging.info("Processing batch synchronously (size=%d)", len(texts))
            if mode == "start":
                return [self.cut_text_start_by_regex(text) for text in texts]
            elif mode == "cut":
                return [self.cut_text_by_regex(text) for text in texts]
            else:
                raise ValueError("Invalid mode. Use 'start' or 'cut'.")
        else:
            logging.info("Processing batch asynchronously (size=%d)", len(texts))
            return asyncio.run(self.process_batch_async(texts, mode=mode))


cutter = TextCutter()
texts = ["Opis stanu faktycznego: ... Pouczenie: ..."] * 20

# Hybrid method with logging and timing
results = cutter.process_batch(texts, mode="start", threshold=10)
print(results[:3])



import time
import asyncio
import functools

class Timing:
    def __init__(self, log_to_file=None, warning_log=None, threshold=None):
        """
        log_to_file: Optional path to a file for all timings.
        warning_log: Optional path to a file for warnings only.
        threshold: Optional time in seconds; if exceeded, a warning is logged.
        """
        self.log_to_file = log_to_file
        self.warning_log = warning_log
        self.threshold = threshold

    def __call__(self, func):
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                self._report(func.__name__, elapsed)
                return result
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start_time
                self._report(func.__name__, elapsed)
                return result
            return sync_wrapper

    def _report(self, name, elapsed):
        message = f"Function '{name}' executed in {elapsed:.4f} seconds"
        print(message)

        # Log all timings if file provided
        if self.log_to_file:
            with open(self.log_to_file, "a") as f:
                f.write(message + "\n")

        # Handle threshold warnings
        if self.threshold and elapsed > self.threshold:
            warning_msg = f"{message} ⚠️ WARNING: Exceeded threshold of {self.threshold:.4f} seconds"
            print(warning_msg)
            if self.warning_log:
                with open(self.warning_log, "a") as f:
                    f.write(warning_msg + "\n")


@Timing(log_to_file="timings.log", warning_log="warnings.log", threshold=0.5)
def compute(n):
    return sum(range(n))

compute(10_000_000)

@Timing(log_to_file="timings.log", warning_log="warnings.log", threshold=1.0)
async def async_compute(n):
    await asyncio.sleep(1)
    return sum(range(n))

asyncio.run(async_compute(1_000_000



def build_prompt(template: str, inputs: dict) -> str:
    """
    Build a prompt by replacing placeholders in the template with user input.

    Args:
        template (str): Prompt template with placeholders like {param1}, {param2}.
        inputs (dict): Dictionary of user inputs where keys match placeholder names.

    Returns:
        str: Final prompt ready for use.
    """
    result = template
    for key, val in inputs.items():
        result = result.replace(f"{{{key}}}", val)
    return result


# Exam