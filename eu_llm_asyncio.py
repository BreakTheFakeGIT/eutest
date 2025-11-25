import random
import argparse
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json
import os

#import ollama
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import product

from src.prompts.prompt_taxes import tax_prompts
from src.utils.text_splitter import process_cut_text, text_splitting
from src.utils.text_process import TextEuJson, TextCleaner
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

# Choose a strong multilingual embedding for Polish
EMBED_MODELS = [("sentence-transformers/paraphrase-multilingual-mpnet-base-v2",768)]
# LLM models available in Ollama (ensure they are pulled locally)
LLM_MODELS = ["hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0", "hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0"]  # rename to your local tags if needed

MODELS = list(product(LLM_MODELS, EMBED_MODELS, repeat=1))
#MODELS = [(x, y, z) for x, (y, z) in MODELS]

# Concurrency controls
MAX_PARALLEL_LLM_CALLS = 6   # tune per machine
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
        slowa_kluczowe_wartosc_eu
    FROM public.interpretacje AS ta
    WHERE 1=1
        AND kategoria_informacji = 1
        AND szablonid IN (1,2)
        AND EXISTS(SELECT 1 FROM public.interpretacje_podobne AS tb WHERE ta.id_informacji = tb.id_informacji_powiazane)
    """

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
    return int(time.perf_counter() * 1000)

# -----------------------------
# Prompt template for LLMs
# -----------------------------
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{prompt_tax}"),
    ("user",
     "Fragment tekstu (wniosku):\n{context}\n\n"
     "Pytanie:\n{question}\n\n"
     "Podaj odpowiedź krótko. Jeśli nie znasz odpowiedzi, napisz krótko: 'Brak informacji'"
    ),
])

def build_llms() -> Dict[str, ChatOllama]:
    """Return LLM clients for Ollama models; synchronous clients wrapped later."""
    return {name: ChatOllama(model=name, temperature=0.0) for name in LLM_MODELS}

def build_embed() -> Dict[str, SentenceTransformer]:
    """Return LLM clients for Ollama models; synchronous clients wrapped later."""
    return {name: SentenceTransformer(model=name, device="cuda") for name in EMBED_MODELS}


# -----------------------------
# Chunking
# -----------------------------
def process_chunk_text(text: str,
               chunk_size: List[int] = [2000],
               chunk_overlap: List[int] = [1000]) -> List[str]:
    text = process_cut_text(text=text)
    results = text_splitting(text=text, chunk_sizes= chunk_size, chunk_overlaps=chunk_overlap)
    return results

def process_topic_text(text: str) -> str:
    text = TextEuJson(text=text).process().to_string()
    text = TextCleaner(text=text).process().to_string()
    return text




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
    INSERT INTO pytodp_wyniki (typ_podatku, id_informacji, fragment, pytanie, odpowiedz, model_llm, fragment_id, eksperyment_klucz, opoznienie_ms, fragment_dlg, pytanie_dlg, odpowiedz_dlg)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (tax_type, text_id, chunk_text_, question, answer, model_name, chunk_id, experiment_key, latency_ms, chunk_text_len, question_len, answer_len))
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
# Core async processing
# -----------------------------
class TaxRAGPipeline:
    def __init__(self):
        self.llms = build_llms()
        #self.embedder = SentenceTransformer(f"{PATH_MODELS}/sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device="cuda")
        #self.retriever = Retriever(self.embedder)
        self.llm_sem = asyncio.Semaphore(MAX_PARALLEL_LLM_CALLS)

    async def llm_answer(self, model_name: str, prompt_tax: str, context: str, question: str) -> Tuple[str, int]:
        """Call LLM synchronously in a thread; measure latency (ms)."""
        prompt = QA_PROMPT.format(prompt_tax=prompt_tax, context=context, question=question)
        start = now_ms()

        def _invoke():
            # ChatOllama has .invoke(); returns ChatMessage or string depending on version
            res = self.llms[model_name].invoke(prompt)
            #print(res)
            # Normalize to string:
            return getattr(res, "content", str(res))

        answer = await asyncio.to_thread(_invoke)
        #print(answer)
        latency = now_ms() - start
        return answer.lower().strip(), latency

    async def embed_answer(self, text: str) -> List[float]:
        return await asyncio.to_thread(lambda: self.embedder.encode(text, normalize_embeddings=True).tolist())

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

        text_id = context_row["id_informacji"]
        tax_type = context_row["typ_podatku"]
        body = context_row["tresc_interesariusz"]
        #dt_w = context_row["dt_wyd"]
        #syg = context_row["syg"]
        #topic = context_row["teza"]

        key_words = ', '.join(context_row["slowa_kluczowe_wartosc_eu"])
        print(key_words)


        # 3) Chunking
        t_chunk_start = time.perf_counter()
        chunks = process_chunk_text(body)
        timings["fragmentacja_s"] = time.perf_counter() - t_chunk_start

        #topic = process_topic_text(text=topic)


        prompt_tax, questions = tax_prompts(tax_type=tax_type)
        if key_words:
            q0 = [f"Do podanych słów: {key_words}, wybierz najlepiej dopasowane zdania z tekstu?"]
            questions = q0 + questions

        ask_llm_embed = list(product(questions, MODELS,  repeat=1))
        results_count = 0
        # Pre-open a DB connection for this context to amortize
        async with await psycopg.AsyncConnection.connect(PG_CONN_STR) as conn:
            # 4) For each chunk and each question, query both LLMs
            tasks = []
            for experiment_key in chunks.keys():
                for chunk_id, chunk_text in enumerate(chunks[experiment_key]):
                    for q, model in ask_llm_embed:
                        model_name, embed = model
                        emb_nm, emb_dim = embed
                        #print(f"Processing text_id={text_id} | tax_type={tax_type} | experiment_key={experiment_key} | chunk_id={chunk_id} | chunk_len={len(chunk_text_)} | model={model_name} | embed_name={emb_nm} | embed_dim={emb_dim} | question={q[:30]}...")
                        # Limit parallelism
                        async def one_call(chunk_id=chunk_id, prompt_tax=prompt_tax, chunk_text_=chunk_text, q=q, model_name=model_name):
                            async with self.llm_sem:
                                answer, latency_ms = await self.llm_answer(model_name=model_name, prompt_tax=prompt_tax, context=chunk_text, question=q)
                            # 6a) Save QA result
                            print(answer)
                            qa_id = await insert_qa_result(conn, tax_type=tax_type,
                                                            text_id=text_id,
                                                            chunk_text_=chunk_text,
                                                            question=q,
                                                            answer=answer,
                                                            model_name=model_name,
                                                            chunk_id=chunk_id,
                                                            experiment_key=experiment_key,
                                                            latency_ms=latency_ms,
                                                            chunk_text_len=len(chunk_text),
                                                            question_len=len(q),
                                                            answer_len= len(answer)
                                                                )
                            
                            # 4b/6b) Vectorize answer + save embedding
                            # vec = await self.embed_answer(answer)
                            # await insert_answer_embedding(conn, qa_id, tax_type, vec)
                            return 1
                        tasks.append(one_call())

            # 5) Measure total LLM phase
            t_llm_start = time.perf_counter()
            # # Write in bursts to reduce contention
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
# Batch orchestrator (1, 7)
# -----------------------------
async def process_batch(limit: int = 5, tax_type: str | None = None) -> Dict[str, Any]:
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


# ---------- CLI ----------
# def parse_args():
#     parser = argparse.ArgumentParser(description="Process questions with Ollama and store embeddings.")
#     #parser.add_argument("--conn-str", default=POSTGRES_DSN, required=True, help="PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)")
#     #parser.add_argument("--json", required=True, help="Path to export JSON file")
#     parser.add_argument("--batch-size-sql", type=int, default=100, help="Batch size for SQL requests")
#     parser.add_argument("--max-retries-sql", type=int, default=3, help="Max retries for SQL requests")
#     parser.add_argument("--model", default="hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0", help="Ollama model name")
#     parser.add_argument("--embedding-model", default="/dane/models/all-MiniLM-L6-v2", help="SentenceTransformer model")
#     print(parser.parse_args())
#     return parser.parse_args()

# ---------- Main ----------
async def main():
    summary = await process_batch(limit=20, tax_type='vat')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    #results = await fetch_texts_batch_fetchall(sql=query_sql, limit= 10, tax_type = None)
    
    #process_context(context_row = results) -> Dict[str, Any]
    #logger.info(f"Processing fetch_texts_batch_fetchall {len(results)} answers completed.")
    #print(results)
    ##results = await fetch_texts_batch_fetchmany(sql=query_sql, limit= 1000, tax_type = None, batch_size_sql=25)
    #logger.info(f"Processing fetch_texts_batch_fetchman {len(results)} completed.")
    #print(results)
    # args = parse_args()
    # #print(args.conn_str)
    # await stream_texts_to_llm_async(query=query_sql,
    #                                 conn_str=POSTGRES_DSN,
    #                                 batch_size_sql=args.batch_size_sql,
    #                                 max_retries_sql=args.max_retries_sql,
    #                                 model_name=args.model,
    #                                 embedding_model=args.embedding_model
    #                                 )
    




if __name__ == "__main__":
    asyncio.run(main())



# # ---------- Ollama Batch Fetch ----------
# async def fetch_batch(prompt: str, questions: List[str], model_name: str, retries=3, backoff_factor=2) -> List[str]:
#     #prompt = """Jesteś ekspertem podatkowym w {type_tax}. Wnioskodawca wprowadził tekst: {text}.\nOdpowiedz na pytania do tekstu jasno i zrozumiale:\n"""
#     for i, q in enumerate(questions, start=1):
#         prompt += f"{i}. {q}\n"
#     print(prompt)

#     for attempt in range(retries):
#         try:
#             response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.2})
#             if not response or "message" not in response or "content" not in response["message"]:
#                 raise ValueError("Invalid response format")
#             print(response["message"])
#             raw_text = response["message"]["content"].strip("\n")
#             print(raw_text)
#             answers = [ans.strip() for ans in raw_text.split("\n") if ans.strip()]
#             print(f"Extracted {len(answers)} answers from response.")
#             #print(answers)
#             while len(answers) < len(questions):
#                 answers.append("")
#             del response, prompt
#             return answers[:len(questions)]

#         except Exception as e:
#             wait_time = backoff_factor ** attempt + random.uniform(0, 1)
#             logger.warning(f"Error fetching batch: {e}. Retrying in {wait_time:.2f}s...")
#             await asyncio.sleep(wait_time)
    
#     logger.error("Failed to fetch batch after retries.")
#     return ["[Error retrieving answer after retries]"] * len(questions)

# # ---------- Async Embedding ----------
# async def async_embed(texts: List[str], embedder) -> List[List[float]]:
#     return await asyncio.to_thread(embedder.encode, texts)




# ---------- Main Pipeline ----------
# async def process_questions(
#     prompt: str,
#     questions: List[str],
#     model_name: str,
#     embedding_model: str
# ):

#     embedder = SentenceTransformer(embedding_model, device='cuda')
#     results = {}
#     try:
#         answers = await fetch_batch(prompt, questions, model_name)
#         logger.info(f"Processing {len(answers)} answers completed.")
#         embeddings = await async_embed(answers, embedder)
#         logger.info(f"Processing {len(embeddings)} embeddings completed.")
#         for q, ans, emb in zip(questions, answers, embeddings):
#             #print(f"Q: {q}\nA: {ans}\nEmbedding snippet: {emb[:5]}...\n{'-'*40}")
#             results[q] = {"answer": ans, "embedding": emb}
#         return results
#     except Exception as e:
#         logger.error(f"Error processing questions: {e}")

    #await save_to_postgres(results, conn_str)




# ---------- Prompts ----------
# async def stream_texts_to_llm_async(query: str,
#                                     conn_str: str,
#                                     batch_size_sql: int,
#                                     max_retries_sql: int,
#                                     model_name: str,
#                                     embedding_model: str
#                                     ):
    
    # async with await psycopg.AsyncConnection.connect(conn_str) as conn:
    #     async with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
    #         await cur.execute(query)
    #         while True:
    #             rows = await cur.fetchmany(batch_size_sql)
    #             if not rows:
    #                 break
    #             texts = [row.get('tresc_interesariusz', []) for row in rows]
    #             id_infos = [row.get('id_informacji', []) for row in rows]
    #             tax_types = [row.get('typ_podatku', []) for row in rows]

    #             for id_info, tax_type, text in zip(id_infos,tax_types,texts):
    #                 for attempt in range(1, max_retries_sql + 1):
    #                     try:
    #                         text = split_text_by_header_regex(text=text)
    #                         results_splits = text_splitting(text=text, chunk_sizes= [800], chunk_overlaps=[200])
    #                         if results_splits:
    #                             first_key = list(results_splits.keys())[0]
    #                             print(f"\nSample result for '{first_key}':")
    #                             for i, text in enumerate(results_splits[first_key][:2]): # Show first 2 chunks
    #                                 print(f"  Chunk {i+1}: '{text[:1000]}...'")
    #                                 prompt_tax, questions_tax = tax_prompts(tax_type=tax_type, user_text=text)
    #                                 print("=" * 50)
    #                                 print(f"Processing id_info: {id_info} | tax_type: {tax_type} | length: {len(text)} | text snippet: {text[:100]}")
    #                                 print("\n" + "=" * 50)



    #                         break
    #                     except Exception as e:
    #                         logger.error(f"Async Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
    #                         if attempt < max_retries_sql:
    #                             delay = 2 ** attempt
    #                             logger.warning(f"Retrying in {delay}s...")
    #                             await asyncio.sleep(delay)
    #                         else:
    #                             logger.error(f"Skipping text after {max_retries_sql} attempts: {text[:50]}")








    # metadata = {
    #     "timestamp": datetime.utcnow().isoformat(),
    #     "batch_size": str(batch_size),
    #     "ollama_model": model_name,
    #     "embedding_model": embedding_model,
    #     "question_count": str(len(questions))
    # }

# python final_script.py \
#   --dsn "postgresql://user:password@localhost:5432/mydb" \
#   --json "qa_embeddings.json" \
#   --batch-size 5 \
#   --model "llama2" \
#   --embedding-model "all-MiniLM-L6-v2"

# CREATE TABLE qa_embeddings (
#     id SERIAL PRIMARY KEY,
#     question TEXT NOT NULL,
#     answer TEXT NOT NULL,
#     embedding FLOAT8[] NOT NULL
# );
# ---------- PostgreSQL Insert ----------
# async def save_to_postgres(results: Dict[str, Dict[str, object]], conn_str: str):
#     try:
#         async with await psycopg.AsyncConnection.connect(conn_str) as conn:
#             async with conn.cursor() as cur:
#                 for question, data in results.items():
#                     await cur.execute(
#                         """
#                         INSERT INTO public.qa_embeddings (question, answer, embedding)
#                         VALUES (%s, %s, %s)
#                         """,
#                         (question, data["answer"], data["embedding"])
#                     )
#             await conn.commit()
#         logger.info("Data saved to PostgreSQL.")
#     except Exception as e:
#         logger.error(f"Database error: {e}")