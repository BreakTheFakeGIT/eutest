
import asyncio
import time
from typing import List, Dict, Any, Tuple
import json
import os

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.prompts.prompt_taxes import tax_prompts
from src.utils.text_splitter import process_cut_text, text_splitting, split_text_by_header_regex

import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


# ---------- Logging Setup ----------
logger = logger_utils.setup_logger(name="eu_llm_asyncio2")


# -----------------------------
# Configuration
# -----------------------------
PG_DSN = os.environ.get("POSTGRES_DSN")
POOL_MIN = 1
POOL_MAX = 10

# Choose a strong multilingual embedding for Polish
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_DIM = 768

# LLM models available in Ollama (ensure they are pulled locally)
LLM_MODELS = ["bielik", "plumm"]  # rename to your local tags if needed

# Concurrency controls
MAX_PARALLEL_LLM_CALLS = 6   # tune per machine
MAX_PARALLEL_DB_WRITES = 10

# -----------------------------
# Helpers
# -----------------------------
def to_pgvector_literal(vec: List[float]) -> str:
    """Convert a list of floats to pgvector textual literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def now_ms() -> int:
    return int(time.perf_counter() * 1000)

# -----------------------------
# Async DB pool
# -----------------------------
# pool = AsyncConnectionPool(
#     conninfo=PG_DSN,
#     min_size=POOL_MIN,
#     max_size=POOL_MAX,
#     open=True,  # open immediately
# )

# -----------------------------
# Tax-type question bank (10 per type)
# -----------------------------
def questions_for_tax_type(tax_type: str) -> List[str]:
    pass


# -----------------------------
# Prompt template for LLMs
# -----------------------------
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{prompt_tax}"),
    ("user",
     "Kontekst (fragment):\n{context}\n\n"
     "Pytanie:\n{question}\n\n"
     "Podaj odpowiedź w 1-3 zdaniach, zwięźle. Jeśli brak danych, powiedz: 'Brak kontekstu.'"
    ),
])

def build_llms() -> Dict[str, ChatOllama]:
    """Return LLM clients for Ollama models; synchronous clients wrapped later."""
    return {name: ChatOllama(model=name, temperature=0.0) for name in LLM_MODELS}

# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str,
               chunk_size: int = 1200,
               chunk_overlap: int = 120) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)







# -----------------------------
# Data access
# -----------------------------
async def fetch_texts_batch(limit: int = 10, tax_type: str | None = None) -> List[Dict[str, Any]]:
    sql = """SELECT
                    tresc_interesariusz,
                    id AS interpretacje_id,
                    id_informacji,
                    typ_podatku
                FROM public.interpretacje AS ta
                WHERE 1=1
                    AND kategoria_informacji = 1
                    AND szablonid IN (1,2)
                    AND EXISTS(SELECT 1 FROM public.interpretacje_podobne AS tb WHERE ta.id_informacji = tb.id_informacji_powiazane)
                ORDER BY id_informacji
                LIMIT 5"""
    params: Tuple[Any, ...] = ()
    if tax_type:
        sql += " WHERE tax_type = %s"
        params = (tax_type,)
    sql += " ORDER BY id DESC LIMIT %s"
    params = params + (limit,)

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return rows


async def fetch_texts_batch(limit: int = 10, tax_type: str | None = None) -> List[Dict[str, Any]]:
    sql = "SELECT id, tax_type, source_uri, body FROM tax_texts"
    params: Tuple[Any, ...] = ()
    if tax_type:
        sql += " WHERE tax_type = %s"
        params = (tax_type,)
    sql += " ORDER BY id DESC LIMIT %s"
    params = params + (limit,)

    async with pool.connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return rows






async def insert_qa_result(conn: psycopg.AsyncConnection,
                           text_id: int, chunk_id: int, question: str,
                           model_name: str, answer: str, latency_ms: int) -> int:
    sql = """
    INSERT INTO qa_results (text_id, chunk_id, question, model_name, answer, latency_ms)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (text_id, chunk_id, question, model_name, answer, latency_ms))
        new_id = (await cur.fetchone())[0]
        return new_id

async def insert_answer_embedding(conn: psycopg.AsyncConnection,
                                  qa_result_id: int, tax_type: str, vec: List[float]) -> None:
    sql = """
    INSERT INTO answer_embeddings (qa_result_id, tax_type, embedding)
    VALUES (%s, %s, %s::vector)
    """
    literal = to_pgvector_literal(vec)
    async with conn.cursor() as cur:
        await cur.execute(sql, (qa_result_id, tax_type, literal))

# -----------------------------
# RAG retrieval (select data)
# -----------------------------
class Retriever:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    async def embed_text(self, text: str) -> List[float]:
        # SentenceTransformer is sync; run off the event loop
        return await asyncio.to_thread(lambda: self.embedder.encode(text, normalize_embeddings=True).tolist())

    async def search_answers(self, query: str, top_k: int = 10,
                             tax_type: str | None = None) -> List[Dict[str, Any]]:
        query_vec = await self.embed_text(query)
        literal = to_pgvector_literal(query_vec)

        where = "WHERE true"
        params: Tuple[Any, ...] = ()
        if tax_type:
            where += " AND tax_type = %s"
            params = (tax_type,)

        sql = f"""
        SELECT qa.id AS qa_result_id,
               qa.text_id,
               qa.chunk_id,
               qa.question,
               qa.model_name,
               qa.answer,
               ae.tax_type,
               ae.embedding <-> %s::vector AS distance
        FROM qa_results qa
        JOIN answer_embeddings ae ON ae.qa_result_id = qa.id
        {where}
        ORDER BY ae.embedding <-> %s::vector
        LIMIT %s
        """
        # we pass the vector twice due to ORDER BY
        params = params + (literal, literal, top_k)

        async with pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, params)
                return await cur.fetchall()

# -----------------------------
# Core async processing
# -----------------------------
class TaxRAGPipeline:
    def __init__(self):
        self.llms = build_llms()
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.retriever = Retriever(self.embedder)
        self.llm_sem = asyncio.Semaphore(MAX_PARALLEL_LLM_CALLS)

    async def llm_answer(self, model_name: str, context: str, question: str) -> Tuple[str, int]:
        """Call LLM synchronously in a thread; measure latency (ms)."""
        prompt = QA_PROMPT.format(context=context, question=question)
        start = now_ms()

        def _invoke():
            # ChatOllama has .invoke(); returns ChatMessage or string depending on version
            res = self.llms[model_name].invoke(prompt)
            # Normalize to string:
            return getattr(res, "content", str(res))

        answer = await asyncio.to_thread(_invoke)
        latency = now_ms() - start
        return answer.strip(), latency

    async def embed_answer(self, text: str) -> List[float]:
        return await asyncio.to_thread(lambda: self.embedder.encode(text, normalize_embeddings=True).tolist())

    async def process_context(self, context_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single raw text:
          - assign 10 questions by tax_type
          - chunk text
          - for each chunk × question × model: ask, embed, save
          - measure timings
        """
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        text_id = context_row["id"]
        tax_type = context_row["tax_type"]
        body = context_row["body"]
        #questions = questions_for_tax_type(tax_type)
        prompt_tax, questions = tax_prompts(tax_type=tax_type)

        # 3) Chunking
        t_chunk_start = time.perf_counter()
        chunks = chunk_text(body)
        timings["chunking_s"] = time.perf_counter() - t_chunk_start

        results_count = 0

        # Pre-open a DB connection for this context to amortize
        async with pool.connection() as conn:
            # 4) For each chunk and each question, query both LLMs
            tasks = []
            for chunk_id, chunk_text_ in enumerate(chunks):
                for q in questions:
                    for model_name in LLM_MODELS:
                        # Limit parallelism
                        async def one_call(chunk_id=chunk_id, chunk_text_=chunk_text_, q=q, model_name=model_name):
                            async with self.llm_sem:
                                answer, latency_ms = await self.llm_answer(model_name, chunk_text_, q)
                            # 6a) Save QA result
                            qa_id = await insert_qa_result(conn, text_id, chunk_id, q, model_name, answer, latency_ms)
                            # 4b/6b) Vectorize answer + save embedding
                            vec = await self.embed_answer(answer)
                            await insert_answer_embedding(conn, qa_id, tax_type, vec)
                            return 1

                        tasks.append(one_call())

            # 5) Measure total LLM phase
            t_llm_start = time.perf_counter()
            # Write in bursts to reduce contention
            done = await asyncio.gather(*tasks)
            timings["qa_llm_s"] = time.perf_counter() - t_llm_start
            results_count = sum(done)

        timings["total_s"] = time.perf_counter() - t0
        return {
            "text_id": text_id,
            "tax_type": tax_type,
            "chunks": len(chunks),
            "questions_per_context": len(questions),
            "models": LLM_MODELS,
            "qa_results_saved": results_count,
            "timings": timings,
        }

    # 8) RAG: public retrieval helper
    async def retrieve(self, query: str, top_k: int = 10, tax_type: str | None = None) -> List[Dict[str, Any]]:
        return await self.retriever.search_answers(query, top_k=top_k, tax_type=tax_type)

# -----------------------------
# Batch orchestrator (1, 7)
# -----------------------------
async def process_batch(limit: int = 5, tax_type: str | None = None) -> Dict[str, Any]:
    """
    1) Download batch of texts from DB
    7) For each context, we have 10 questions (questions_for_tax_type)
    """
    pipeline = TaxRAGPipeline()
    t_fetch_start = time.perf_counter()
    rows = await fetch_texts_batch(limit=limit, tax_type=tax_type)
    t_fetch = time.perf_counter() - t_fetch_start

    if not rows:
        return {"fetched": 0, "timings": {"fetch_s": t_fetch}, "contexts": []}

    # Process contexts in parallel
    contexts_tasks = [pipeline.process_context(r) for r in rows]
    t_proc_start = time.perf_counter()
    contexts_results = await asyncio.gather(*contexts_tasks)
    t_proc = time.perf_counter() - t_proc_start

    return {
        "fetched": len(rows),
        "timings": {"fetch_s": t_fetch, "process_s": t_proc},
        "contexts": contexts_results,
    }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":

    print(QA_PROMPT)
    # async def main():
    #     summary = await process_batch(limit=3, tax_type="VAT")
    #     print(json.dumps(summary, ensure_ascii=False, indent=2))

    #     # RAG query example:
    #     pipeline = TaxRAGPipeline()
    #     hits = await pipeline.retrieve("Stawki VAT dla usług elektronicznych", top_k=5, tax_type="VAT")
    #     print("\nTop-5 RAG hits:")
    #     for h in hits:
    #         print(f"- QA#{h['qa_result_id']} (model={h['model_name']}, dist={h['distance']:.4f})")
    #         print(f"  Q: {h['question']}\n  A: {h['answer'][:200]}...\n")

    # asyncio.run(main())
