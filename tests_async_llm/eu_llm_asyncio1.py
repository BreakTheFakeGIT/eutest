import os
import asyncio
import time
import ollama
import psycopg
from psycopg import sql
import random
import argparse
from typing import List, Dict, Any, Tuple
import psycopg.rows
from sentence_transformers import SentenceTransformer
from datetime import datetime
from itertools import product
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import ChatPromptTemplate
from langchain_ollama import ChatOllama
from src.utils.text_cleaner import pipeline_process, extract_fact_q
from src.prompts.prompt_taxes import get_questions, get_prompt
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
SCHEMA_NAME = 'public'
POOL_MIN = 1
POOL_MAX = 10

# Embedded models
#"sdadas/st-polish-paraphrase-from-mpnet","all-MiniLM-L6-v2"
EMBED_MODELS = ['sdadas/mmlw-retrieval-roberta-large-v2', 'sdadas/stella-pl-retrieval']
# LLM models available in Ollama (ensure they are pulled locally)
LLM_MODELS = ["hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0", "hf.co/NikolayKozloff/Llama-PLLuM-8B-instruct-Q8_0-GGUF:Q8_0"]


MODELS = LLM_MODELS #list(product(LLM_MODELS,T_EMBED_MODELS, repeat=1))

# Concurrency controls
MAX_PARALLEL_LLM_CALLS = 10  # tune per machine
MAX_PARALLEL_DB_WRITES = 10
# ---------- SQL Query ----------
query_sql = sql.SQL("""SELECT DISTINCT
                    id_informacji,
                    typ_podatku,
                    tresc_interesariusz,
                    dt_wyd,
                    syg,
                    teza,
                    slowa_kluczowe_wartosc_eu
                FROM public.interpretacje AS ta
                WHERE 1=1
                    AND kategoria_informacji = 1
                    AND szablonid IN (1,2)
                    AND NOT EXISTS (
                        SELECT 1
                        FROM public.qa_results AS qa 
                        WHERE qa.interp_id = ta.id_informacji) 
                ORDER BY id_informacji DESC
                LIMIT 10;""")


# -----------------------------
# Helpers
# -----------------------------
def async_connection(conn_str: str=PG_CONN_STR):
    asynconnect = psycopg.AsyncConnection.connect(conn_str)
    return asynconnect


def to_pgvector_literal(vec: List[float]) -> str:
    """Convert a list of floats to pgvector textual literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def now_ms() -> int:
    """Return current time in milliseconds."""
    return int(time.perf_counter() * 1000)


def now_s() -> int:
    """Return current time in seconds."""
    return int(time.perf_counter() * 1000000)

# -----------------------------
# Prompt template for LLMs
# -----------------------------
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system}"),
    ("user",
    "Fragment wniosku \n{context}\n\n"
    "Pytanie \n{question}\n\n"
    "W puntkach odpowiedź na pytanie bazując wyłącznie na powyższym fragmencie wniosku. "
    "Odpowidaj krótko i zrozumiale. "
    "Nie proponuj. "
    "Pomiń pytania w odpowiedzi. "
    "Jeśli nie znasz odpowiedzi, napisz: ' brak informacji '"
    ),
])

# -----------------------------
# LLM Clients
# -----------------------------
def rows_collect(rows: dict) -> Tuple[List[int], List[str], List[str], List[Dict[str, Any]]]:
    texts = [pipeline_process(text=row.get('tresc_interesariusz', [])) for row in rows]
    interp_ids = [row.get('id_informacji', []) for row in rows]
    tax_types = [row.get('typ_podatku', []) for row in rows]
    topic = [x[1] for x in [pipeline_process(row.get("teza", [])) for row in rows]]
    file_reference =  [row.get('syg', []) for row in rows] 
    release_date = [row.get("dt_wyd", []) for row in rows]
    keywords = [', '.join(row.get("slowa_kluczowe_wartosc_eu", [])) for row in rows]
    interp = [{'id_informacji': interp_ids,
                'typ_podatku': tax_types,
                'teza': topic,
                'sygnatura': file_reference,
                'data_wydania': release_date,
                'slowa_kluczowe': keywords} for _ in rows]
    system_prompt = [get_prompt(area=tax_type) for tax_type in tax_types]
    questions = [get_questions(area=tax_type) for tax_type in tax_types]
    chunks = [(text_start_end, text_norm) for (text_start_end,text_norm) in texts]
    unzipped_rows = list(zip(interp_ids, tax_types, texts, interp, system_prompt, chunks, questions))
    return unzipped_rows


# -----------------------------
# PostgreSQL Inserts
# -----------------------------

async def insert_qa_results(conn: psycopg.AsyncConnection,
                            tax_type: str,
                            interp_id: int,
                            chunk_cnt: int,
                            chunk_id: int,
                            chunk_text: str,
                            question_id: int,
                            question: str,
                            answer: str,
                            latency_ms: Dict[str, float],
                            experiment: Dict[str, Any],
                            special_code: int,
                            created_at: datetime = logger_utils.set_datetime_local()
                           ) -> int:

    sql = """
    INSERT INTO public.qa_results (tax_type, interp_id, chunk_cnt, chunk_id, chunk_text, question_id, question, answer, latency_ms, experiment, special_code, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """

    try:
        async with conn.cursor() as cur:
            await cur.execute(sql, (tax_type, interp_id, chunk_cnt, chunk_id, chunk_text, question_id, question, answer, latency_ms, experiment, special_code, created_at))
            new_id = (await cur.fetchone())[0]
            return new_id
    except Exception as e:
        logger.error(f"Error inserting QA result for interp_id={interp_id}, tax_type={tax_type}, question={question}, chunk_id={chunk_id}: {e}")
        raise





# -----------------------------------
# ---------- Text Chunking ----------
# -----------------------------------
def text_chunking(text: str) -> List[str]:
    """
    Process a single raw text
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1750,
                                            chunk_overlap=100,
                                            length_function=len
                                            )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if len(chunk)>50]

def proces_text_chunking(text: str, questions: List[str]) -> List[Tuple]:
    chunks = text_chunking(text=text)
    chunks = chunks[:3]
    #prompt_tax, questions = tax_prompts(tax_type=tax_type)
    results = [(chunk_id, question_id, chunk_text, question) for (chunk_id, chunk_text), (question_id, question) in product(enumerate(chunks,start=1), enumerate(questions, start=1)) ]
    return results

class TaxPipeline:
    def __init__(self, llm_model_name: str):
        self.llm_model_name = llm_model_name
        self.llms =  ChatOllama(model=self.llm_model_name, temperature=0)
        #self.asyncembeddingmanager = AsyncEmbeddingManager(add_text_to_list(words=EMBED_MODELS, prefix=PATH_MODELS, suffix=''))
        self.llm_sem = asyncio.Semaphore(MAX_PARALLEL_LLM_CALLS)

    async def llm_answer(self, system_prompt: str, context: str, question: str) -> Tuple[str, int]:
        prompt = QA_PROMPT.format(system=system_prompt, context=context, question=question)
        start = now_ms()
        def _invoke():
            res = self.llms.invoke(prompt)
            return getattr(res, "content", str(res))

        answer = await asyncio.to_thread(_invoke)
        latency = now_ms() - start
        return answer.strip(), latency

    async def process_context(self, row_context: List) -> Dict[str, Any]:
        """
        Process a single raw text:
        """
        timings: Dict[str, float] = {}
        results_count = 0
        t_total_start  = now_ms()
        interp_id, tax_type, text, interp, system_prompt, chunks, questions = row_context
        print(f'=== iterp_id: {interp_id}, tax_type: {tax_type} ===')
        #text_start_end, text_norm = text
        _ , text_norm = text
        list_proces_text_chunking = proces_text_chunking(text=text_norm, questions=questions)

        tasks = []
        async with await psycopg.AsyncConnection.connect(PG_CONN_STR) as conn:
            for chunk_id, question_id, chunk_text, question in list_proces_text_chunking:
                print(f'START=== iterp_id: {interp_id}, tax_type: {tax_type}, question: {question},len() {len(list_proces_text_chunking)}, chunk_id: {chunk_id} question_id: {question_id} {system_prompt}')
                async def one_call(system_prompt=system_prompt, chunk_text=chunk_text, question=question):
                    async with self.llm_sem:
                        answer, latency_ms = await self.llm_answer(system_prompt=system_prompt, context=chunk_text, question=question)
                        timings['llm_latency_ms'] = latency_ms
                        print(answer)

                    qa_id = await insert_qa_results(
                                conn=conn,
                                tax_type=tax_type,
                                interp_id=interp_id,
                                chunk_cnt=0,
                                chunk_id=chunk_id,
                                chunk_text=chunk_text,
                                question_id=question_id,
                                question=question,
                                answer=answer,
                                latency_ms=timings,
                                experiment={'chunk_size': 1750, 'chunk_overlap': 100, 'llm_model_name': self.llm_model_name},
                                special_code=0,
                            )
                    print(qa_id)


                        #print(f'END=== iterp_id: {interp_id}, tax_type: {tax_type}, question: {question} latency_ms {latency_ms} ===')

                    return 1
                tasks.append(one_call())


        t_llm_start = now_ms()
        done = await asyncio.gather(*tasks)
        timings["gather_tasks_llm_ms"] = now_ms() - t_llm_start
        results_count = sum(done)
        timings["total_ms"] = now_ms() - t_total_start

        return {
            "timings": timings,
            "qa_results_saved": results_count,

        }

# ---------- Prompts ----------
async def process_batch(query: str,
                        conn_str: str,
                        batch_size_sql: int,
                        #max_retries_sql: int,
                        llm_model_name: str,
                        # embedding_model: str
                        ):
    pipeline = TaxPipeline(llm_model_name=llm_model_name)
    async with await psycopg.AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor(name="stream_cursor", row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(query)
            while True:
                rows = await cur.fetchmany(batch_size_sql)
                if not rows:
                    break
                unzipped_rows = rows_collect(rows=rows)

                contexts_tasks = [pipeline.process_context(r) for r in unzipped_rows]
                t_proc_start = time.perf_counter()
                contexts_results = await asyncio.gather(*contexts_tasks)
                print(contexts_results)
                t_proc = time.perf_counter() - t_proc_start


    return {
        "fetched": len(rows),
    }



#hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0
# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Process questions with Ollama and store embeddings.")
    #parser.add_argument("--conn-str", default=POSTGRES_DSN, required=True, help="PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)")
    #parser.add_argument("--json", required=True, help="Path to export JSON file")
    parser.add_argument("--batch-size-sql", type=int, default=100, help="Batch size for SQL requests")
    parser.add_argument("--max-retries-sql", type=int, default=3, help="Max retries for SQL requests")
    parser.add_argument("--llm-model-name", default="hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0", help="Ollama model name")
    parser.add_argument("--embedding-model", default="/dane/models/sdadas/stella-pl-retrieval", help="SentenceTransformer model")
    print(parser.parse_args())
    return parser.parse_args()

# ---------- Main ----------
async def main():
    args = parse_args()
    await process_batch(query=query_sql,
                                    conn_str=PG_CONN_STR,
                                    batch_size_sql=args.batch_size_sql,
                                    #max_retries_sql=args.max_retries_sql,
                                    llm_model_name=args.llm_model_name
                                    )



if __name__ == "__main__":
    asyncio.run(main())

############################################################################
                # print(([len(b) for (_,b) in chunks]))
                # print([len(b[0]) for (_,b) in chunks])
                # print(type(zipped_rows))
                #print(zipped_rows)
#print(list_proces_text_chunking)



                            # # Chunking example
                            # text_start_end, text_norm = text
                            # chunks = chunk_text(text=text_norm, chunk_size=1750, chunk_overlap=250)
                            # prompt_tax, questions = tax_prompts(tax_type=tax_type)
                            # results = [(chunk_id, question_id, chunk_text, question) for (chunk_id, chunk_text), (question_id, question) in product(enumerate(chunks,start=1), enumerate(questions, start=1)) ]

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

                            #print("Chunking example results:", results)
                            # print(f"Normalized text length: {len(text_norm)}, {format(text_norm[:50])}...")
                            # text_norm_fact, text_norm_q = extract_fact_q(text=text_norm)
                            # # if text_norm_q:
                            # #     _, text_q = extract_fact_q(text=text_start_end)
                            # #     list_of_questions = [q.lower() for q  in text_q.split('\n') if len(q) > 50]
                            # #     if list_of_questions:
                            # #         #zip(['pytania']*len(list_of_questions), list_of_questions)
                            # #         print("="*10+' START TEXT '+"="*50)
                            # #         print(list_of_questions)
                            # #         print("="*10+' END TEXT '+"="*50)
                            # #         # TODO embeddings
                            # #         # TODO zip
                            # #         # TODO save to postgres

                            # Example: plan for a text of length 1000, with chunk_size=200 and overlap=50
                            # len_text = len(text_norm)
                            # chunk_size_lst = [x for x in [1750,1250,750,300]]
                            # p_overlap_from_chunk_size_lst = [0.1,0.2,0.4]
                            # overlap_lst = [int(round(p_cs*p_ofcs,0)) for (p_cs,p_ofcs) in list(product(chunk_size_lst, p_overlap_from_chunk_size_lst, repeat=1))]

                            # for (chunk_size, overlap) in product(chunk_size_lst, overlap_lst, repeat=1):
                            #     try:
                        
                            #         if overlap >= chunk_size:
                            #             continue
                            #         planner = ChunkPlanner(length=len_text, chunk_size=chunk_size, overlap=overlap)
                            #         print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
                            #         n_chunks = planner.num_chunks()
                            #         print("N =", n_chunks)
                            #         # print("Indices:", planner.indices())
                            #         # Cmax = ChunkPlanner.max_chunk_size(length=len_text, overlap=overlap, max_chunks=n_chunks)
                            #         # print(f"Max chunk size for {n_chunks} chunks with overlap {overlap}: {Cmax}")
                            #         # Max overlap allowed for N_max
                            #         print(planner.split_text(text=text_norm))
                            #         # Which overlaps yield exactly N=7?
                            #         # overlaps = ChunkPlanner.overlaps_for_exact_N(length=len_text, chunk_size=chunk_size, N=n_chunks)
                            #         # print(f"Overlaps for exactly N={n_chunks}: {overlaps}")
                            #     except Exception as e:
                            #         logger.error(f"Error in chunk planning with chunk_size={chunk_size}, overlap={overlap}: {e}")
                            #         continue


                            # overlap = 50
                            # planner = ChunkPlanner(length=len_text, chunk_size=1750, overlap=750)
                            # planner.num_chunks_static(length: int, chunk_size: int, overlap: int) 
                            # print("N =", planner.num_chunks())
                            # print("Indices:", planner.indices())
# async def generate_answer(
#     chunk_text: str,
#     question: str,
#     llm: ChatOllama,
#     semaphore: asyncio.Semaphore
# ) -> Dict[str, Any]:
#     """
#     Generates an answer for a specific chunk and question.
#     Uses a semaphore to limit concurrent calls to the local Ollama instance.
#     """
#     async with semaphore:
#         prompt = ChatPromptTemplate.from_template(
#             "Kontekst:\n{context}\n\nPytanie: {question}\n\nOdpowiedź (zwięźle):"
#         )
#         chain = prompt | llm
        
#         try:
#             # Run the chain asynchronously
#             response = await chain.ainvoke({"context": chunk_text, "question": question})
#             return {
#                 "question": question,
#                 "answer": response.content,
#                 "chunk_text": chunk_text,
#                 "success": True
#             }
#         except Exception as e:
#             print(f"Error generating answer: {e}")
#             return {"success": False}
# def product_chunking(tax_type: str, text: str) -> List[Tuple]:
#     """
#     Process a single raw text
#     """
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1750,
#                                             chunk_overlap=250,
#                                             length_function=len,
#                                             separators=["\n\n", "\n", "?"],
#                                             )

#     chunks = [chunk for chunk in splitter.split_text(text) if len(chunk)>50]
#     questions = get_questions(area=tax_type)
#     product_chunks = [(chunk_id, question_id, chunk_text, question) for (chunk_id, chunk_text), (question_id, question) in product(enumerate(chunks,start=1), enumerate(questions, start=1)) ]
#     return product_chunks