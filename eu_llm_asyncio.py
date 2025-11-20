import os
import asyncio
import ollama
import psycopg
from psycopg import sql
import random
import argparse
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer

from src.prompts.prompt_taxes import tax_prompts
from src.utils.text_splitter import process_cut_text
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


# ---------- Logging Setup ----------
logger = logger_utils.setup_logger(name="eu_llm_asyncio")

###################################
TBL_INTERP = os.environ.get("TBL_INTERP")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

# ---------- SQL Query ----------
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



# ---------- Ollama Batch Fetch ----------
async def fetch_batch(prompt: str, questions: List[str], model_name: str, retries=3, backoff_factor=2) -> List[str]:
    #prompt = """Jesteś ekspertem podatkowym w {type_tax}. Wnioskodawca wprowadził tekst: {text}.\nOdpowiedz na pytania do tekstu jasno i zrozumiale:\n"""
    for i, q in enumerate(questions, start=1):
        prompt += f"{i}. {q}\n"

    for attempt in range(retries):
        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}], options={"temperature": 0.3})
            if not response or "message" not in response or "content" not in response["message"]:
                raise ValueError("Invalid response format")

            raw_text = response["message"]["content"].strip()
            print(raw_text)
            answers = [ans.strip() for ans in raw_text.split("\n") if ans.strip()]
            # while len(answers) < len(questions):
            #     answers.append("")
            del response, prompt
            return answers[:len(questions)]

        except Exception as e:
            wait_time = backoff_factor ** attempt + random.uniform(0, 1)
            logger.warning(f"Error fetching batch: {e}. Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
    
    logger.error("Failed to fetch batch after retries.")
    return ["[Error retrieving answer after retries]"] * len(questions)

# ---------- Async Embedding ----------
async def async_embed(texts: List[str], embedder) -> List[List[float]]:
    return await asyncio.to_thread(embedder.encode, texts)




# ---------- Main Pipeline ----------
async def process_questions(
    prompt: str,
    questions: List[str],
    model_name: str,
    embedding_model: str
):

    embedder = SentenceTransformer(embedding_model, device='cuda')
    results = {}
    try:
        answers = await fetch_batch(prompt, questions, model_name)
        embeddings = await async_embed(answers, embedder)
        for q, ans, emb in zip(questions, answers, embeddings):
            results[q] = {"answer": ans, "embedding": emb}
        return results
    except Exception as e:
        logger.error(f"Error processing questions: {e}")

    #await save_to_postgres(results, conn_str)




# ---------- Prompts ----------
async def stream_texts_to_llm_async(query: str,
                                    conn_str: str,
                                    batch_size_sql: int,
                                    max_retries_sql: int,
                                    model_name: str,
                                    embedding_model: str
                                    ):
    async with await psycopg.AsyncConnection.connect(conn_str) as conn:
        async with conn.cursor(name="stream_cursor", row_factory=psycopg.rows.dict_row) as cur:
            await cur.execute(query)
            while True:
                rows = await cur.fetchmany(batch_size_sql)
                if not rows:
                    break
                texts = [process_cut_text(text=row.get('tresc_interesariusz', [])) for row in rows]
                id_infos = [row.get('id_informacji', []) for row in rows]
                tax_types = [row.get('typ_podatku', []) for row in rows]

                for id_info, tax_type, text in zip(id_infos,tax_types,texts):
                    for attempt in range(1, max_retries_sql + 1):
                        try:
                            prompt_tax, questions_tax = tax_prompts(tax_type=tax_type, user_text=text)
                            print("=" * 50)
                            print(f"Processing id_info: {id_info} | tax_type: {tax_type} | length: {len(text)} | text snippet: {text[:100]}")
                            print("\n" + "=" * 50)
                            results = await process_questions(
                                prompt=prompt_tax,
                                questions=questions_tax,
                                model_name=model_name,
                                embedding_model=embedding_model
                                )
                            print(results)
                            #await save_to_postgres(results, conn_str=conn)
                            break
                        except Exception as e:
                            logger.error(f"Async Error streaming LLM for text: {text[:50]} | Attempt {attempt} | {e}")
                            if attempt < max_retries_sql:
                                delay = 2 ** attempt
                                logger.warning(f"Retrying in {delay}s...")
                                await asyncio.sleep(delay)
                            else:
                                logger.error(f"Skipping text after {max_retries_sql} attempts: {text[:50]}")



# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Process questions with Ollama and store embeddings.")
    #parser.add_argument("--conn-str", default=POSTGRES_DSN, required=True, help="PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)")
    #parser.add_argument("--json", required=True, help="Path to export JSON file")
    parser.add_argument("--batch-size-sql", type=int, default=100, help="Batch size for SQL requests")
    parser.add_argument("--max-retries-sql", type=int, default=3, help="Max retries for SQL requests")
    parser.add_argument("--model", default="hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0", help="Ollama model name")
    parser.add_argument("--embedding-model", default="/dane/models/all-MiniLM-L6-v2", help="SentenceTransformer model")
    print(parser.parse_args())
    return parser.parse_args()

# ---------- Main ----------
async def main():
    args = parse_args()
    #print(args.conn_str)
    await stream_texts_to_llm_async(query=query_sql,
                                    conn_str=POSTGRES_DSN,
                                    batch_size_sql=args.batch_size_sql,
                                    max_retries_sql=args.max_retries_sql,
                                    model_name=args.model,
                                    embedding_model=args.embedding_model
                                    )



if __name__ == "__main__":
    asyncio.run(main())


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