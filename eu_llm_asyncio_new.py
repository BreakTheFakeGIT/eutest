import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Union
import json
import os
import torch
import re

#import ollama
import psycopg
from psycopg.rows import dict_row

from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from itertools import product
from collections import Counter

from src.prompts.prompt_taxes import tax_prompts
from src.utils.text_splitter import process_chunk_text
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


# ---------- Logging Setup ----------
logger = logger_utils.setup_logger(name="eu_llm_asyncio")

# -----------------------------
# Configuration
# -----------------------------
TBL_INTERP = os.environ.get("TBL_INTERP")
PG_CONN_STR = os.environ.get("POSTGRES_DSN")
PATH_MODELS = os.environ.get("MODELS_HF")
POOL_MIN = 1
POOL_MAX = 10

# Embedded models
EMBED_MODELS = ["sdadas/st-polish-paraphrase-from-mpnet",
                "all-MiniLM-L6-v2"]
# LLM models available in Ollama (ensure they are pulled locally)
LLM_MODELS = ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0", 
              "hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0"]  # rename to your local tags if needed


MODELS = LLM_MODELS #list(product(LLM_MODELS,T_EMBED_MODELS, repeat=1))

# Concurrency controls
MAX_PARALLEL_LLM_CALLS = 6  # tune per machine
MAX_PARALLEL_DB_WRITES = 10

# -----------------------------
# SQL Query
# -----------------------------
query_sql = """SELECT
        id_informacji,
        typ_podatku,
        tresc_interesariusz,
        CAST(dt_wyd AS date) dt_wyd,
        syg,
        teza,
        slowa_kluczowe_wartosc_eu,
        wartosc_eu
    FROM public.interpretacje AS ta
    WHERE 1=1
        AND kategoria_informacji = 1
        AND szablonid IN (1,2)
        AND EXISTS(SELECT 1 FROM public.interpretacje_podobne AS tb WHERE ta.id_informacji = tb.id_informacji_powiazane)
    """
# TODO ADD to WHERE OR EXISTS(SELECT 1 FROM prompt_odp AS tb WHERE ta.id_informacji = tb.id_informacji)
# NOTE REMOVE EXISTS(SELECT 1 FROM public.interpretacje_podobne ...) when you will start for all.

async def fetch_texts_batch_fetchall(sql: str, limit: int = 10, tax_type: str | None = None) -> List[Dict[str, Any]]:
    params: Tuple[Any, ...] = ()
    if tax_type:
        sql += " AND typ_podatku = %s"
        params = (tax_type,)
    sql += " ORDER BY id_informacji DESC LIMIT %s"
    params = params + (limit,)

    async with await psycopg.AsyncConnection.connect(PG_CONN_STR) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            logger.info(f"Processing fetch_texts_batch_fetchall {len(rows)} completed.")
            return rows

# -----------------------------
# Helpers
# -----------------------------

def to_pgvector_literal(vec: List[float]) -> str:
    """Convert a list of floats to pgvector textual literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def now_ms() -> int:
    """Return current time in milliseconds."""
    return int(time.perf_counter() * 1000)

def add_text_to_list(words, prefix="", suffix=""):
    """Adds a prefix and suffix to each word in the list. """
    return [f"{prefix}{word}{suffix}" for word in words]

def generate_combinations(list1: List[str], list2: List[str]) -> List[Tuple]:
    """ Generate all combinations of questions and models with repeat=1."""
    return list(product(list1, list2, repeat=1))

# Compile the regex pattern once for efficiency

_pattern = re.compile(
    r'^\s*(?:\w+\s*:\s*)?(?:brak informacji|nie dotyczy)[\.\:]*\s*$',
    re.IGNORECASE
)
def is_empty_or_special(answer: str) -> bool:
    return bool(_pattern.match(answer or ""))

# -----------------------------
# Prompt template for LLMs
# -----------------------------
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{prompt_tax}"),
    ("user",
     "Fragment tekstu (wniosku):\n{context}\n\n"
     "Pytanie:\n{question}\n\n"
     "Odpowidaj krótko i zrozumiale z tekstu. Nie proponuj. Pomiń pytania w odpowiedzi. Jeśli nie znasz odpowiedzi, napisz: 'brak informacji'."
    ),
])

# -----------------------------
# Async Embedding Manager
# -----------------------------

class AsyncEmbeddingManager:
    def __init__(self, model_names: List[str]):
        """
        Initialize multiple SentenceTransformer models.
        Automatically uses GPU if available.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models: Dict[str, SentenceTransformer] = {
            name: SentenceTransformer(name, device=self.device) for name in model_names
        }

    async def embed(self, model_name: str, text: Union[str, List[str]]) -> List[List[float]]:
        """ Async embedding for a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.models[model_name].encode, text)
        return embeddings.tolist() if isinstance(embeddings, list) else [embeddings.tolist()]

    async def embed_all(self, text: Union[str, List[str]]) -> Dict[str, List[List[float]]]:
        """
        Embed text using all models concurrently.
        Returns a dict: {model_name: embeddings}
        """
        tasks = [self.embed(name, text) for name in self.models.keys()]
        results = await asyncio.gather(*tasks)
        return dict(zip(self.models.keys(), results))

    def list_models(self) -> List[str]:
        """Return names of all loaded models."""
        return list(self.models.keys())


# -----------------------------
# LLM Clients
# -----------------------------

def build_llms() -> Dict[str, ChatOllama]:
    """Return LLM clients for Ollama models; synchronous clients wrapped later."""
    return {name: ChatOllama(model=name, temperature=0.0) for name in LLM_MODELS}

# -----------------------------
# PostgreSQL Inserts
# -----------------------------

async def insert_qa_result(conn: psycopg.AsyncConnection,
                           text_id: int,
                           tax_type: str,
                           experiment_key: str,
                           chunk_id: int,
                           chunk_text_: str,
                           question: str,
                           model_name: str,
                           answer: str,
                           latency_ms: int,
                           chunk_text_len: int = 0,
                           question_len: int = 0,
                           answer_len: int = 0
                           ) -> int:
    sql = """
    INSERT INTO prompt_odp (typ_podatku, id_informacji, fragment, pytanie, odpowiedz, model_llm, fragment_id, eksperyment_klucz, opoznienie_ms, fragment_dlg, pytanie_dlg, odpowiedz_dlg)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
    """
    try: 
        async with conn.cursor() as cur:
            await cur.execute(sql, (tax_type, text_id, chunk_text_, question, answer, model_name, chunk_id, experiment_key, latency_ms, chunk_text_len, question_len, answer_len))
            new_id = (await cur.fetchone())[0]
            return new_id
    except Exception as e:
        logger.error(f"Error inserting QA result for text_id={text_id}, tax_type={tax_type}, , question={question}, model={model_name}, chunk_id={chunk_id}: {e}")
        raise

# -----------------------------
# Insert answer embedding
# -----------------------------

async def insert_answer_embedding768(conn: psycopg.AsyncConnection,
                                  tax_type: str,
                                  tax_id: int,
                                  topic: str,
                                  signature: str,
                                  release_date: datetime,
                                  keywords: List[str],
                                  qa_result_id: int,
                                  model_embedding: str,
                                  embedding: List[float]
                                  ) -> int:
    sql = """
    INSERT INTO odp_embedding768 (typ_podatku, id_informacji, teza, sygnatura, data_wydania, slowa_kluczowe, prompt_odp_id, model_embedding, embedding)
    VALUES (%s, %s, %s,%s, %s, %s, %s, %s, %s::vector)
    """
    embedding = to_pgvector_literal(embedding)
    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, (tax_type, tax_id, topic, signature, release_date, keywords, qa_result_id, model_embedding, embedding))
    except Exception as e:
        logger.error(f"Error inserting answer embedding768 for qa_result_id={qa_result_id}, tax_type={tax_type}, tax_id={tax_id}, model_embedding={model_embedding}, em: {e}")
        raise




# -----------------------------
# Core async processing
# -----------------------------

class TaxRAGPipeline:
    def __init__(self):
        self.llms = build_llms()
        self.asyncembeddingmanager = AsyncEmbeddingManager(add_text_to_list(words=EMBED_MODELS, prefix=PATH_MODELS, suffix=''))
        self.llm_sem = asyncio.Semaphore(MAX_PARALLEL_LLM_CALLS)

    async def llm_answer(self, model_name: str, prompt_tax: str, context: str, question: str) -> Tuple[str, int]:
        """Call LLM synchronously in a thread; measure latency (ms)."""
        prompt = QA_PROMPT.format(prompt_tax=prompt_tax, context=context, question=question)
        start = now_ms()

        def _invoke():
            # ChatOllama has .invoke(); returns ChatMessage or string depending on version
            res = self.llms[model_name].invoke(prompt)
            # Normalize to string:
            return getattr(res, "content", str(res))

        answer = await asyncio.to_thread(_invoke)
        latency = now_ms() - start
        return answer.lower().strip(), latency

    async def process_context(self, context_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single raw text:
          - assign questions by tax_type
          - chunk text
          - for each chunk × question × model: ask, embed, save
          - measure timings
        """
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()
        text_id, tax_type, text, key_words = context_row["id_informacji"], context_row["typ_podatku"], context_row["tresc_interesariusz"], context_row["slowa_kluczowe_wartosc_eu"]

        # Chunking
        t_chunk_start = time.perf_counter()
        chunks = process_chunk_text(text=text, chunk_size=[2000], chunk_overlap=[400])
        timings["fragmentacja_s"] = time.perf_counter() - t_chunk_start

        # Prepare questions
        prompt_tax, questions = tax_prompts(tax_type=tax_type)
        if key_words:
            q0 = add_text_to_list(words=key_words, prefix='Do podanego słowa: "', suffix='", wybierz najlepiej dopasowane jedno zdanie z tekstu?')
            questions += list(set(q0))

        results_count = 0
        # Pre-open a DB connection for this context to amortize
        async with await psycopg.AsyncConnection.connect(PG_CONN_STR) as conn:
            tasks = []
            for q in questions:
                for experiment_key in chunks.keys():
                    for chunk_id, chunk_text in enumerate(chunks[experiment_key]):
                        async def one_call(chunk_id=chunk_id,
                                           prompt_tax=prompt_tax,
                                           chunk_text=chunk_text,
                                           q=q):
                            async with self.llm_sem:
                                answer0, latency_ms = await self.llm_answer(model_name=LLM_MODELS[0], prompt_tax=prompt_tax, context=chunk_text, question=q)
                                answer1, latency_ms = await self.llm_answer(model_name=LLM_MODELS[1], prompt_tax=prompt_tax, context=chunk_text, question=q)
                            print(f"text_id={text_id}, tax_type={tax_type}, q='{q}'")
                            print(LLM_MODELS[0])
                            print(answer0)
                            print(LLM_MODELS[1])
                            print(answer1)


                            # Split by numbered points (e.g., "1. ", "2. ")
                            parts = re.split(r'\d+\.\s*', text)

                            # Remove empty strings and print results
                            parts = [p.strip() for p in parts if p.strip()]
                            for i, part in enumerate(parts, start=1):
                                print(f"{i}: {part}")

                            # if not answer or is_empty_or_special(answer=answer):
                            #     print(" ".format(model_name,))
                            #     return 0  # Skip empty answers
                            # print(f"text_id={text_id}, tax_type={tax_type}, q='{q}', model={model_name}")
                            # print(answer)
                            # Save QA result
                            # qa_id = await insert_qa_result(conn,
                            #                                 tax_type=tax_type,
                            #                                 text_id=text_id,
                            #                                 chunk_text=chunk_text,
                            #                                 question=q,
                            #                                 answer=answer,
                            #                                 model_name=model_name,
                            #                                 chunk_id=chunk_id,
                            #                                 experiment_key=experiment_key,
                            #                                 latency_ms=latency_ms,
                            #                                 text_len=text_len, ##
                            #                                 chunk_text_len=len(chunk_text),
                            #                                 question_len=len(q),
                            #                                 answer_len= len(answer),
                            #                                 chunk_text_words = Counter(chunk_text),
                            #                                 answer_words = Counter(chunk_text)

                            #                                 )


                            # list_models = self.asyncembeddingmanager.list_models()
                            # vec = await self.asyncembeddingmanager.embed(model_name=list_models[0], text=answer)
                            # await insert_answer_embedding768(conn,
                            #       tax_type=tax_type,
                            #       tax_id=text_id,
                            #       topic=context_row["teza"],
                            #       signature=context_row["syg"],
                            #       release_date=context_row["dt_wyd"],
                            #       keywords=key_words,
                            #       qa_result_id=qa_id,
                            #       # TODO model_name=model_name,
                            #       model_embedding= list_models[0],
                            #       embedding=vec[0]
                            #       )

                            return 1
                        tasks.append(one_call())

            # Measure total LLM phase
            t_llm_start = time.perf_counter()
            # Write in bursts to reduce contention
            done = await asyncio.gather(*tasks)
            timings["qa_llm_s"] = time.perf_counter() - t_llm_start
            results_count = sum(done)

        timings["total_s"] = time.perf_counter() - t0
        return {
            "id_informacji": text_id,
            "typ_podatku": tax_type,
            "qa_results_saved": results_count,
            "timings": timings,
        }


# -----------------------------
# Batch orchestrator
# -----------------------------
async def process_batch(limit: int = 25, tax_type: str | None = None) -> Dict[str, Any]:
    """
    1) Download batch of texts from DB
    7) For each context, we have 10 questions (questions_for_tax_type)
    """
    pipeline = TaxRAGPipeline()
    t_fetch_start = time.perf_counter()
    rows = await fetch_texts_batch_fetchall(limit=limit, tax_type=tax_type, sql=query_sql)
    t_fetch = time.perf_counter() - t_fetch_start

    if not rows:
        return {"fetched": 0, "timings": {"fetch_s": t_fetch}, "contexts": []}

    # Process contexts in parallel
    contexts_tasks = [pipeline.process_context(r) for r in rows]
    t_proc_start = time.perf_counter()
    contexts_results = await asyncio.gather(*contexts_tasks)
    print(contexts_results)
    t_proc = time.perf_counter() - t_proc_start

    return {
        "fetched": len(rows),
        "timings": {"fetch_s": t_fetch, "process_s": t_proc},
        "contexts": contexts_results,
    }


# ---------- Main ----------
async def main():
    summary = await process_batch(limit=10, tax_type=None)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


#  async def one_call(chunk_id=chunk_id, prompt_tax=prompt_tax, chunk_text=chunk_text, q=q, model_name=model_name):
#                             async with self.llm_sem:
#                                 answer, latency_ms = await self.llm_answer(model_name=model_name, prompt_tax=prompt_tax, context=chunk_text, question=q)
#                             if not answer or answer.strip().lower() in ('brak informacji', 'nie dotyczy'):
#                                 return 0  # Skip empty answers
#                             print(answer)
#                             # Save QA result
#                             qa_id = await insert_qa_result(conn,
#                                                             tax_type=tax_type,
#                                                             text_id=text_id,
#                                                             chunk_text_=chunk_text,
#                                                             question=q,
#                                                             answer=answer,
#                                                             model_name=model_name,
#                                                             chunk_id=chunk_id,
#                                                             experiment_key=experiment_key,
#                                                             latency_ms=latency_ms,
#                                                             chunk_text_len=len(chunk_text),
#                                                             question_len=len(q),
#                                                             answer_len= len(answer)
#                                                             )
#                             print(f"Inserted QA result id={qa_id} for text_id={text_id}, tax_type={tax_type}, model={model_name}, chunk_id={chunk_id}.")


#                             list_models = self.asyncembeddingmanager.list_models()
#                             vec = await self.asyncembeddingmanager.embed(model_name=list_models[0], text=answer)
#                             await insert_answer_embedding768(conn,
#                                   tax_type=tax_type,
#                                   tax_id=text_id,
#                                   topic=context_row["teza"],
#                                   signature=context_row["syg"],
#                                   release_date=context_row["dt_wyd"],
#                                   keywords=key_words,
#                                   qa_result_id=qa_id,
#                                   model_embedding= list_models[0],
#                                   embedding=vec[0]
#                                   )

#                             return 1