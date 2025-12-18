import asyncio
import time
from typing import List, Dict, Any, Tuple
import json
import os
import re
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.text_cleaner import pipeline_process
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
EMBED_MODEL_NAME = '/dane/models/sdadas/stella-pl-retrieval'
EMBED_DIM = 1024
LLM_MODELS = ['hf.co/SpeakLeash/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0']  # rename to your local tags if needed

# Concurrency controls
MAX_PARALLEL_LLM_CALLS = 10 # tune per machine

SCHEMA_NAME = 'public'
TABLE_QA_RESULTS = 'qa_results_test'
TABLE_ANSWER_EMBEDDINGS_QA = 'answer_embeddings_qa_test'
TABLE_STAT_RESULTS = 'stat_results_test'

CHUNK_SIZE = 1750
CHUNK_OVERLAP = 175

# -----------------------------
# Helpers
# -----------------------------
def to_pgvector_literal(vec: List[float]) -> str:
    """Convert a list of floats to pgvector textual literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def now_ms() -> int:
    return int(time.perf_counter() * 1000)

def is_empty_or_special(text: str) -> int:
    phrases = ['brak',
               '**brak',
               'brak informacji',
               '**brak informacji**',
               'brak odniesienia',
               'nie ma wystarczających',
               'nie można',
               'nie jest możliwe',
               'nie dotyczy',
               'nie posiadam',
               'nie ma informacji',
               'wnioskodawca nie korzysta',
               'nie została wskazana',
               'nie wskazała'
               ]
    pattern = r'\b(?:' + '|'.join(map(re.escape, phrases)) + r')\b'
    if re.search(pattern, text, re.IGNORECASE):
        return 1
    else:
        return 0


# Prompt template for LLMs
# -----------------------------
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("user",
     "Fragment wniosku \n{context}\n\n"
     "Pytanie \n{question}\n\n"
     "Odpowidaj krótko, rzeczowo i pełnym zdaniem, bazując wyłącznie na powyższym fragmencie wniosku. "
     "Nie sugeruj, nie dopowiadaj oraz pomiń pytania w odpowiedzi."
     "Jeśli nie znasz odpowiedzi napisz krótko: ' brak informacji ' "
    ),
])


def build_llms() -> Dict[str, ChatOllama]:
    """Return LLM clients for Ollama models; synchronous clients wrapped later."""
    return {name: ChatOllama(model=name, temperature=0.0) for name in LLM_MODELS}

# -----------------------------
# Chunking
# -----------------------------
def splitter_chunk_text(text: str,
               chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    chunks = [chunk for chunk in chunks if len(chunk)>50]
    return chunks[:4]

# -----------------------------
# Data access
# -----------------------------
def split_into_batches(data_list: List[Dict], batch_size: int):
    """
    Splits a list into smaller batches of specified size.
    """
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]

async def fetch_texts_batch(limit: int=10,
                            tax_type: str | None=None
                            ) -> List[Dict[str, Any]]:

    query_sql = """
    SELECT 
        id_informacji,
        typ_podatku,
        tresc_interesariusz,
        TO_CHAR(dt_wyd, 'YYYY-MM-DD') AS dt_wyd,
        syg,
        teza,
        slowa_kluczowe_wartosc_eu,
        wartosc_eu
    FROM public.interpretacje AS ta
    WHERE True
        AND kategoria_informacji = 1
        AND szablonid IN (1,2)
        AND typ_podatku IN ('vat','pit','cit','pcc','psd',
        'akcyza','op','gry','malpki','spw','pt','pkop',
        'spdet','fin','cukier','wip','globe','nip','inne')
        AND NOT EXISTS (
            SELECT 1
            FROM public.qa_results AS qa 
            WHERE qa.text_id = ta.id_informacji
        )
        AND NOT EXISTS (
            SELECT 1
            FROM public.stat_results AS qb
            WHERE qb.text_id = ta.id_informacji
        )
    """
    params: Tuple[Any, ...] = ()
    if tax_type:
        query_sql += " AND typ_podatku = %s"
        params = (tax_type,)

    query_sql += " ORDER BY id_informacji DESC LIMIT %s"
    params = params + (limit,)

    async with await psycopg.AsyncConnection.connect(PG_DSN, autocommit=True) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(query_sql, params)
            rows = await cur.fetchall()
            return rows

async def insert_qa_results(conn: psycopg.AsyncConnection,
                            schema_name: str,
                            table_name: str,
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
                            llm_latency_ms: int,
                            keywords: List[str]
                        ) -> int:

    query_sql = sql.SQL("""INSERT INTO {} (text_id, tax_type, chunk_id, question_id, model_id, model_name, chunk_text, question, answer, is_excluded, llm_latency_ms, keywords)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id;""").format(sql.Identifier(schema_name, table_name)).as_string()

    async with conn.cursor() as cur:
        await cur.execute(query_sql, (text_id, tax_type, chunk_id, question_id, model_id, model_name, chunk_text, question, answer, is_excluded, llm_latency_ms, keywords))
        new_id = (await cur.fetchone())[0]
        return new_id

async def insert_answer_embeddings(conn: psycopg.AsyncConnection,
                            schema_name: str,
                            table_name: str,
                            qa_results_id: int,
                            text_id: int,
                            tax_type: str,
                            question_id: int,
                            show_in_chat: Dict[str,Any],
                            is_excluded: int,
                            type_text: str,
                            vector: List[float]
                            ) -> None:

    query_sql = sql.SQL("""INSERT INTO {} (qa_results_id, text_id, tax_type, question_id, show_in_chat, is_excluded, type_text, embedding)
    VALUES (%s, %s,%s, %s,%s, %s, %s,%s::vector)""").format(sql.Identifier(schema_name, table_name)).as_string()

    literal = to_pgvector_literal(vector)
    async with conn.cursor() as cur:
        await cur.execute(query_sql, (qa_results_id, text_id, tax_type, question_id, show_in_chat, is_excluded, type_text, literal))
    await conn.commit()

async def insert_stat_results(conn: psycopg.AsyncConnection,
                            schema_name: str,
                            table_name: str,
                            text_id: int | None,
                            tax_type: str | None,
                            type_results: str,
                            results: Dict[str,Any]
                            ) -> None:

    query_sql = sql.SQL("""INSERT INTO {} (text_id, tax_type, type_results, results)
    VALUES (%s, %s, %s, %s)""").format(sql.Identifier(schema_name, table_name)).as_string()

    async with conn.cursor() as cur:
        await cur.execute(query_sql, (text_id, tax_type, type_results, results))
    await conn.commit()


# -----------------------------
# Core async processing
# -----------------------------
class TaxRAGPipeline:
    def __init__(self):
        self.llms = build_llms()
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME, device='cuda')
        self.embedder.half()
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
        return answer.lower(), latency

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
        keywords = context_row["wartosc_eu"]
        _ , topic =  pipeline_process(context_row["teza"])
        _ , text_norm = pipeline_process(context_row["tresc_interesariusz"])
        show_in_chat = json.dumps({
            'id_informacji': text_id,
            'typ_podatku': tax_type,
            'teza': topic,
            'sygnatura': context_row["syg"],
            'data_wydania': context_row["dt_wyd"],
            'slowa_kluczowe': ','.join(context_row["slowa_kluczowe_wartosc_eu"])
            }, ensure_ascii=False)

        system = get_prompt(area=tax_type)
        questions = get_questions(area=tax_type)

        # Chunking
        t_chunk_start = time.perf_counter()
        chunks = splitter_chunk_text(text=text_norm[:8840])
        timings["chunking_s"] = round(time.perf_counter() - t_chunk_start,3)
        results_count = 0

        # Pre-open a DB connection for this context to amortize
        async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
            # For each chunk and each question, query both LLMs
            tasks = []
            for model_id, model_name in enumerate(LLM_MODELS, start=1):
                for q_id, q in enumerate(questions, start=1):
                    for chunk_id, chunk_text in enumerate(chunks, start=1):
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
                                show_in_chat=show_in_chat,
                                keywords= keywords
                                ):
                            async with self.llm_sem:
                                answer, llm_latency_ms = await self.llm_answer(system=system, model_name=model_name, context=chunk_text, question=question)
                            timings["llm_latency_s"] = round(llm_latency_ms/1000,3)

                            is_excluded = is_empty_or_special(text=answer)
                            if question_id>3 and is_excluded==1:
                                return 0

                            #################################################################
                            # vectorize answer + save embedding
                            t_insert_start = time.perf_counter()
                            answer = answer.replace('**odpowiedź:**','')
                            qa_id = await insert_qa_results(conn=conn,
                                                        schema_name=SCHEMA_NAME,
                                                        table_name=TABLE_QA_RESULTS,
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
                                                        llm_latency_ms=llm_latency_ms,
                                                        keywords=keywords
                                                        )
                            if qa_id:
                                vec_ans = await self.embed_answer(text=answer)
                                await insert_answer_embeddings(conn=conn,
                                                            schema_name=SCHEMA_NAME,
                                                            table_name=TABLE_ANSWER_EMBEDDINGS_QA,
                                                            qa_results_id=qa_id,
                                                            text_id=text_id,
                                                            tax_type=tax_type,
                                                            question_id=question_id,
                                                            show_in_chat=show_in_chat,
                                                            is_excluded=is_excluded,
                                                            type_text='answer',
                                                            vector=vec_ans)
                                del vec_ans
                                if question_id == 1:
                                    vec_chunk = await self.embed_answer(text=chunk_text)
                                    await insert_answer_embeddings(conn=conn,
                                                                schema_name=SCHEMA_NAME,
                                                                table_name=TABLE_ANSWER_EMBEDDINGS_QA,
                                                                qa_results_id=qa_id,
                                                                text_id=text_id,
                                                                tax_type=tax_type,
                                                                question_id=question_id,
                                                                show_in_chat=show_in_chat,
                                                                is_excluded=is_excluded,
                                                                type_text='chunk',
                                                                vector=vec_chunk)
                                    del vec_chunk

                            parts = re.split(r'\d+\.\s|\d+\:\s', answer)
                            answer_split = [p.strip() for p in parts if p.strip()]
                            digit_pattern = re.compile(r'\d')
                            answer_split = [p for p in answer_split if len(p)>3 or not digit_pattern.search(p)]
                            text_type = [f'keywords']*len(answer_split)
                            for ans_split, txt_type in zip(answer_split,text_type):
                                try:
                                    qa_id = await insert_qa_results(conn=conn,
                                                        schema_name=SCHEMA_NAME,
                                                        table_name=TABLE_QA_RESULTS,
                                                        text_id=text_id,
                                                        tax_type=tax_type,
                                                        chunk_id=chunk_id,
                                                        question_id=question_id,
                                                        model_id=model_id,
                                                        model_name=model_name,
                                                        chunk_text=chunk_text,
                                                        question=question,
                                                        answer=ans_split,
                                                        is_excluded=is_excluded,
                                                        llm_latency_ms=llm_latency_ms,
                                                        keywords=keywords
                                                        )
                                    vec_ans_split = await self.embed_answer(text=ans_split)
                                    await insert_answer_embeddings(conn=conn,
                                                                schema_name=SCHEMA_NAME,
                                                                table_name=TABLE_ANSWER_EMBEDDINGS_QA,
                                                                qa_results_id=qa_id,
                                                                text_id=text_id,
                                                                tax_type=tax_type,
                                                                question_id=question_id,
                                                                show_in_chat=show_in_chat,
                                                                is_excluded=is_excluded,
                                                                type_text=txt_type,
                                                                vector=vec_ans_split)
                                except Exception as e:
                                    logger.error(f"Answer_split error: {e}")
                                    continue

                            #################################################################
                            timings["insert_s"] = round(time.perf_counter() - t_insert_start,3)
                            results = {
                                "text_id": text_id,
                                "tax_type": tax_type,
                                "chunk_id": chunk_id,
                                "question_id":question_id,
                                "chunk_text_len": len(chunk_text),
                                "answer_len": len(answer),
                                "text_norm_len": len(text_norm),
                                "questions_per_context": len(questions),
                                "llm_models": LLM_MODELS,
                                "embed_models": EMBED_MODEL_NAME,
                                "embed_dim": EMBED_DIM,
                                "chunk_size": CHUNK_SIZE,
                                "chunk_overlap": CHUNK_OVERLAP,
                                "asyncio_semaphore": MAX_PARALLEL_LLM_CALLS,
                                "set_datetime_local": logger_utils.set_datetime_local(),
                                "timings": timings,
                                } 
                            await insert_stat_results(conn=conn,
                                    schema_name=SCHEMA_NAME,
                                    table_name=TABLE_STAT_RESULTS,
                                    text_id=text_id,
                                    tax_type=tax_type,
                                    type_results='one_call',
                                    results = json.dumps(results, ensure_ascii=False)
                                    )
                            return 1
                        tasks.append(one_call())

            # Measure total LLM phase
            t_llm_start = time.perf_counter()
            # Write in bursts to reduce contention
            done = await asyncio.gather(*tasks)
            timings["qa_llm_s"] = round(time.perf_counter() - t_llm_start,3)
            results_count = sum(done)
            timings["total_s"] = round(time.perf_counter() - t0,3)
            results = {
                "text_id": text_id,
                "tax_type": tax_type,
                "chunks": len(chunks),
                "text_norm_len": len(text_norm),
                "questions_per_context": len(questions),
                "llm_models": LLM_MODELS,
                "embed_models": EMBED_MODEL_NAME,
                "embed_dim": EMBED_DIM,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "asyncio_semaphore": MAX_PARALLEL_LLM_CALLS,
                "qa_results_saved": results_count,
                "set_datetime_local": logger_utils.set_datetime_local(),
                "timings": timings,
            }
            await insert_stat_results(conn=conn,
                                    schema_name=SCHEMA_NAME,
                                    table_name=TABLE_STAT_RESULTS,
                                    text_id=text_id,
                                    tax_type=tax_type,
                                    type_results='process_context',
                                    results = json.dumps(results, ensure_ascii=False)
                                    )
        return results


# -----------------------------
# Batch orchestrator 
# -----------------------------
async def process_batch(limit: int=10, tax_type: str | None=None, batch_limit: int=5) -> Dict[str, Any]:
    """
    Download batch of texts from DB. For each context process
    """
    logger.info(f"Start pipeline.")

    pipeline = TaxRAGPipeline()
    t_fetch_start = time.perf_counter()
    rows = await fetch_texts_batch(limit=limit, tax_type=tax_type)
    t_fetch = round(time.perf_counter() - t_fetch_start,3)
    if not rows:
        return {"fetched": 0, "timings": {"fetch_texts_batch_s": t_fetch}, "contexts": []}

    logger.info(f"Length rows {len(rows)}. Limit {limit}. Tax type: {tax_type}. Type:{type(rows)}. Batch limit {batch_limit}") 

    summary = []
    batches = list(split_into_batches(rows, batch_limit))
    for batch_id, batch in enumerate(batches, start=1):

        try:
            id_list = [r['id_informacji'] for r in batch]
            logger.info(f"Start batch {batch_id}. Text_id (id_informacji): {id_list}.")
            # Process contexts in parallel
            contexts_tasks = [pipeline.process_context(r) for r in batch]
            t_proc_start = time.perf_counter()
            contexts_results = await asyncio.gather(*contexts_tasks)
            t_proc = round(time.perf_counter() - t_proc_start,3)

            results = {
            "fetched": len(batch),
            "text_ids": id_list,
            "timings": {"fetch_texts_batch_s": t_fetch, "contexts_results_s": t_proc},
            "contexts": contexts_results,
            }
            async with await psycopg.AsyncConnection.connect(PG_DSN) as conn:
                await insert_stat_results(conn=conn,
                                        schema_name=SCHEMA_NAME,
                                        table_name=TABLE_STAT_RESULTS,
                                        text_id=0,
                                        tax_type="batch",
                                        type_results='process_batch',
                                        results = json.dumps(results, ensure_ascii=False)
                                        )

            summary.append(results)
            del results, contexts_results, batch
            logger.info(f"End batch {batch_id}. Text_id (id_informacji): {id_list}. ")
        except Exception as e:
            logger.error(f"Batch error: {e}")
            continue
    return summary



# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    async def main():
        summary = await process_batch(limit=30, tax_type=None, batch_limit=10)
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    asyncio.run(main())

