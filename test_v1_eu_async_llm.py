import asyncio
import time
from typing import List, Dict, Any, Tuple
import json
import os
import psycopg
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.text_cleaner import pipeline_process #, extract_fact_q
from src.prompts.prompt_taxes import get_questions, get_prompt
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


# ---------- Logging Setup ----------
logger = logger_utils.setup_logger(name="eu_async_llm")


# -----------------------------
# Configuration
# -----------------------------
PG_DSN = os.environ.get("POSTGRES_DSN")
# /dane/models/sdadas/stella-pl-retrieval
# /dane/models/sdadas/mmlw-retrieval-roberta-large-v2
# Choose a strong multilingual embedding for Polish
EMBED_MODEL_NAME = '/dane/models/sdadas/stella-pl-retrieval'
EMBED_DIM = 1024

# LLM models available in Ollama (ensure they are pulled locally)
# 'hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0'
# 'hf.co/SpeakLeash/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0'
LLM_MODELS = ['hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0']  # rename to your local tags if needed

# Concurrency controls
MAX_PARALLEL_LLM_CALLS = 10   # tune per machine
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
# Tax-type question bank (10 per type)
# -----------------------------
def questions_for_tax_type(tax_type: str) -> List[str]:
    # Customize per your domain; ensuring 10 questions per context
    base = tax_type.lower()
    return [
        f"{base}: W punktach, wymień 10 słów kluczowych dotyczących podatku?",
        f"{base}: W 5 punktach, wymień artykuły, przepisy wskazane w tekście?",
        f"{base}: W 3 zdaniach, podsumuj fragment tekstu?",
        f"{base}: W 1 zdaniu kim jest wnisoksowdawca: osobą fizyczną, jednostką samorządu terytorialnego (gmina, powiat, województwo), spółką, czy innym podmiotem (np. stowarzyszenie, spółdzielnia itp.)?",
        f"{base}: W 1 zdaniu, czy wsnioskodawca korzysta ze zwolnienia, odliczenia lub ulg?"
    ]


# -----------------------------
# Prompt template for LLMs
# -----------------------------
# QA_PROMPT = ChatPromptTemplate.from_messages([
#     ("system", "Jesteś ekspertem podatkowym. Odpowidaj krótko i zrozumiale. Nie proponuj. Pomiń pytania w odpowiedzi."),
#     ("user",
#      "Fragment wniosku: \n{context}\n\n"
#      "Pytanie: \n{question}\n\n"
#      "Wypunktuj odpowiedź na pytanie bazując wyłącznie na powyższym fragmencie wniosku. "
#      "Jeśli nie znasz odpowiedzi, napisz krótko: ' brak informacji '. "
#     ),
# ])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("user",
     "Fragment wniosku \n{context}\n\n"
     "Pytanie \n{question}\n\n"
     "Wypunktuj odpowiedź na pytanie bazując wyłącznie na powyższym fragmencie wniosku. "
     "Odpowidaj krótko i zrozumiale. "
     "Nie proponuj. "
     "Pomiń pytania w odpowiedzi. "
     "Jeśli nie znasz odpowiedzi, napisz: ' brak informacji ' "
    ),
])





def build_llms() -> Dict[str, ChatOllama]:
    """Return LLM clients for Ollama models; synchronous clients wrapped later."""
    return {name: ChatOllama(model=name, temperature=0.0) for name in LLM_MODELS}

# -----------------------------
# Chunking
# -----------------------------
def splitter_chunk_text(text: str,
               chunk_size: int = 1750,
               chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        #separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    chunks = [chunk for chunk in chunks if len(chunk)>50]
    return chunks[:6]


# -----------------------------
# SQL Query
# -----------------------------
query_sql = """
SELECT 
    id_informacji,
    typ_podatku,
    tresc_interesariusz,
    TO_CHAR(dt_wyd, 'YYYY-MM-DD') AS dt_wyd,
    syg,
    teza,
    slowa_kluczowe_wartosc_eu
FROM public.interpretacje AS ta
WHERE True
    AND kategoria_informacji = 1
    AND szablonid IN (1,2)
    AND NOT EXISTS (
        SELECT 1
        FROM public.qa_results AS qa 
        WHERE qa.text_id = ta.id_informacji
    )
"""

# -----------------------------
# Data access
# -----------------------------

async def fetch_texts_batch(sql: str = query_sql, limit: int = 10, tax_type: str | None = None, is_batch: bool = False, id_limit_down: int = 0, id_limit_up: int = 10000000) -> List[Dict[str, Any]]:
    params: Tuple[Any, ...] = ()
    if is_batch:
        sql += " AND id_informacji> %s AND id_informacji <= %s"
        params = (id_limit_down, id_limit_up,)
    if tax_type:
        sql += " AND typ_podatku = %s"
        params = (tax_type,)

    sql += " ORDER BY id_informacji DESC LIMIT %s"
    params = params + (limit,)

    async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
            return rows

#sql = "SELECT id, tax_type, source_uri, body FROM tax_texts" 

async def insert_qa_results(conn: psycopg.AsyncConnection,
                            text_id: int,
                            tax_type: str,
                            chunk_id: int,
                            question_id: int,
                            model_id: int,
                            model_name: str,
                            chunk_text:str,
                            question: str,
                            answer: str,
                            is_excluded: int,
                            llm_latency_ms: int
                        ) -> int:
    sql = """
    INSERT INTO qa_results (text_id, tax_type, chunk_id, question_id, model_id, model_name, chunk_text, question, answer, is_excluded, llm_latency_ms)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (text_id, tax_type, chunk_id, question_id, model_id, model_name, chunk_text, question, answer, is_excluded, llm_latency_ms))
        new_id = (await cur.fetchone())[0]
        return new_id

async def insert_answer_embeddings(conn: psycopg.AsyncConnection,
                                qa_result_id: int, text_id: int, tax_type: str,
                                question_id: int, show_in_chat: Dict[str,Any], is_excluded: int, type_text: str,
                                vec: List[float]) -> None:
    sql = """
    INSERT INTO answer_embeddings (qa_result_id, text_id, tax_type, question_id, show_in_chat, is_excluded, type_text, embedding)
    VALUES (%s, %s,%s, %s,%s, %s, %s,%s::vector)
    """
    literal = to_pgvector_literal(vec)
    async with conn.cursor() as cur:
        await cur.execute(sql, (qa_result_id, text_id, tax_type, question_id, show_in_chat, is_excluded, type_text, literal))


async def insert_stat_results(conn: psycopg.AsyncConnection,
                            type_results: str,
                            results: Dict[str,Any]
                            ) -> None:
    sql = """
    INSERT INTO stat_results (type_results, results)
    VALUES (%s, %s)
    """
    async with conn.cursor() as cur:
        await cur.execute(sql, (type_results, results))



# -----------------------------
# RAG retrieval (select data)
# -----------------------------
class Retriever:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder

    async def embed_text(self, text: str) -> List[float]:
        # SentenceTransformer is sync; run off the event loop
        return await asyncio.to_thread(lambda: self.embedder.encode(text, normalize_embeddings=True, show_progress_bar = False).tolist())

    async def search_answers(self, query: str, top_k: int = 10,
                             tax_type: str | None = None) -> List[Dict[str, Any]]:
        query_vec = await self.embed_text(query)
        literal = to_pgvector_literal(query_vec)

        where = "WHERE true"
        params: Tuple[Any, ...] = ()
        if tax_type:
            where += " AND qa.tax_type = %s "
            params = (tax_type,)

        sql = f"""
        SELECT qa.id,
               qa.text_id,
               qa.chunk_id,
               qa.question,
               qa.model_name,
               qa.answer,
               ae.tax_type,
               ae.embedding::vector <-> %s::vector AS distance
        FROM qa_results AS qa
        JOIN answer_embeddings AS ae ON ae.qa_result_id = qa.id
        {where}
        ORDER BY ae.embedding::vector <-> %s::vector
        LIMIT %s
        """
        # we pass the vector twice due to ORDER BY
        params = params + (literal, literal, top_k)

        async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(sql, params)
                return await cur.fetchall()

# -----------------------------
# Core async processing
# -----------------------------
class TaxRAGPipeline:
    def __init__(self):
        self.llms = build_llms()
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device='cuda')
        self.embedder.half()
        self.retriever = Retriever(self.embedder)
        self.llm_sem = asyncio.Semaphore(MAX_PARALLEL_LLM_CALLS)

    async def llm_answer(self, model_name: str, system: str, context: str, question: str) -> Tuple[str, int]:
        """Call LLM synchronously in a thread; measure latency (ms)."""
        prompt = QA_PROMPT.format(system=system, context=context, question=question)
        start = now_ms()

        def _invoke():
            # ChatOllama has .invoke(); returns ChatMessage or string depending on version
            res = self.llms[model_name].invoke(prompt)
            # Normalize to string:
            return getattr(res, "content", str(res))

        answer = await asyncio.to_thread(_invoke)
        latency = now_ms() - start
        return answer.lower().strip(), latency

    async def embed_answer(self, text: str) -> List[float]:
        return await asyncio.to_thread(lambda: self.embedder.encode(text, normalize_embeddings=True, show_progress_bar = False).tolist())

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

        text_id = context_row["id_informacji"]
        tax_type = context_row["typ_podatku"]
        _ , body = pipeline_process(context_row["tresc_interesariusz"])
        _ , topic =  pipeline_process(context_row["teza"])
        show_in_chat = json.dumps({
            'id_informacji': text_id,
            'typ_podatku': tax_type,
            'teza': topic,
            'sygnatura': context_row["syg"],
            'data_wydania': context_row["dt_wyd"],
            'slowa_kluczowe': ', '.join(context_row["slowa_kluczowe_wartosc_eu"])
            }, ensure_ascii=False)

        #questions = questions_for_tax_type(tax_type)
        system = get_prompt(area=tax_type)
        questions = get_questions(area=tax_type) 

        # 3) Chunking
        t_chunk_start = time.perf_counter()
        chunks = splitter_chunk_text(body)
        timings["chunking_s"] = round(time.perf_counter() - t_chunk_start,3)

        results_count = 0

        # Pre-open a DB connection for this context to amortize
        async with await psycopg.AsyncConnection.connect(PG_DSN)  as conn:
            # 4) For each chunk and each question, query both LLMs
            tasks = []
            for chunk_id, chunk_text in enumerate(chunks, start=1):
                for q_id, q in enumerate(questions, start=1):
                    for model_id, model_name in enumerate(LLM_MODELS, start=1):
                        # Limit parallelism
                        async def one_call(
                                text_id=text_id,
                                tax_type=tax_type,
                                chunk_id=chunk_id,
                                question_id=q_id,
                                model_id=model_id,
                                model_name=model_name,
                                chunk_text=chunk_text,
                                question=q,
                                show_in_chat=show_in_chat
                                ):
                            async with self.llm_sem:
                                answer, llm_latency_ms = await self.llm_answer(system=system, model_name=model_name, context=chunk_text, question=question)
                            timings["llm_latency_s"] = round(llm_latency_ms/1000,3)
                            # 6a) Save QA result
                            is_excluded = 0
                            qa_id = await insert_qa_results(conn=conn,
                                                        text_id=text_id,
                                                        tax_type=tax_type,
                                                        chunk_id=chunk_id,
                                                        question_id=question_id,
                                                        model_id=model_id,
                                                        model_name=model_name,
                                                        chunk_text=chunk_text,
                                                        question=question,
                                                        answer=answer,
                                                        is_excluded=is_excluded,
                                                        llm_latency_ms=llm_latency_ms)
                            if question_id == 1:
                                vec_chunk = await self.embed_answer(chunk_text)
                                await insert_answer_embeddings(conn=conn, 
                                                              qa_result_id=qa_id,
                                                              text_id=text_id,
                                                              tax_type=tax_type,
                                                              question_id=question_id,
                                                              show_in_chat=show_in_chat,
                                                              is_excluded=is_excluded,
                                                              type_text='chunk',
                                                              vec=vec_chunk)
                                del vec_chunk

                            # 4b/6b) Vectorize answer + save embedding
                            vec = await self.embed_answer(answer)
                            await insert_answer_embeddings(conn=conn, 
                                                              qa_result_id=qa_id,
                                                              text_id=text_id,
                                                              tax_type=tax_type,
                                                              question_id=question_id,
                                                              show_in_chat=show_in_chat,
                                                              is_excluded=is_excluded,
                                                              type_text='answer',
                                                              vec=vec)
                            del vec
                            return 1

                        tasks.append(one_call())

            # 5) Measure total LLM phase
            t_llm_start = time.perf_counter()
            # Write in bursts to reduce contention
            done = await asyncio.gather(*tasks)
            timings["qa_llm_s"] = round(time.perf_counter() - t_llm_start,3)
            results_count = sum(done)

        timings["total_s"] = round(time.perf_counter() - t0,3)
        return {
            "text_id": text_id,
            "tax_type": tax_type,
            "chunks": len(chunks),
            "questions_per_context": len(questions),
            "llm_models": LLM_MODELS,
            "embed_models": EMBED_MODEL_NAME,
            "embed_dim": EMBED_DIM,
            "asyncio_semaphore": MAX_PARALLEL_LLM_CALLS,
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
    t_fetch = round(time.perf_counter() - t_fetch_start,3)

    if not rows:
        return {"fetched": 0, "timings": {"fetch_texts_batch_s": t_fetch}, "contexts": []}

    # Process contexts in parallel
    contexts_tasks = [pipeline.process_context(r) for r in rows]
    t_proc_start = time.perf_counter()
    contexts_results = await asyncio.gather(*contexts_tasks)
    t_proc = round(time.perf_counter() - t_proc_start,3)
    return {
        "fetched": len(rows),
        "timings": {"fetch_texts_batch_s": t_fetch, "contexts_results_s": t_proc},
        "contexts": contexts_results,
    }

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    async def main():
        summary = await process_batch(limit=3, tax_type="vat")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
            await insert_stat_results(conn=conn,type_results='process_batch', results = json.dumps(summary, ensure_ascii=False))

        # # RAG query example:
        # pipeline = TaxRAGPipeline()
        # hits = await pipeline.retrieve("Stawki VAT dla usług elektronicznych", top_k=5, tax_type="vat")
        # print("\nTop-5 RAG hits:")
        # for h in hits:
        #     print(f"- QA#{h['qa_result_id']} (model={h['model_name']}, dist={h['distance']:.4f})")
        #     print(f"  Q: {h['question']}\n  A: {h['answer'][:200]}...\n")

    asyncio.run(main())
