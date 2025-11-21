import asyncio
import ollama
import random
import json
import argparse
import logging
from typing import List, Dict
from datetime import datetime
from sentence_transformers import SentenceTransformer
import psycopg

# ---------- Logging Setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------- Ollama Batch Fetch ----------
async def fetch_batch(questions: List[str], model_name="llama2", retries=3, backoff_factor=2) -> List[str]:
    prompt = "Answer the following questions clearly and concisely:\n"
    for i, q in enumerate(questions, start=1):
        prompt += f"{i}. {q}\n"

    for attempt in range(retries):
        try:
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            if not response or "message" not in response or "content" not in response["message"]:
                raise ValueError("Invalid response format")

            raw_text = response["message"]["content"].strip()
            answers = [ans.strip() for ans in raw_text.split("\n") if ans.strip()]
            while len(answers) < len(questions):
                answers.append("[No answer returned]")
            return answers[:len(questions)]

        except Exception as e:
            wait_time = backoff_factor ** attempt + random.uniform(0, 1)
            logging.warning(f"Error fetching batch: {e}. Retrying in {wait_time:.2f}s...")
            await asyncio.sleep(wait_time)
    
    logging.error("Failed to fetch batch after retries.")
    return ["[Error retrieving answer after retries]"] * len(questions)

# ---------- Async Embedding ----------
async def async_embed(texts: List[str], embedder) -> List[List[float]]:
    return await asyncio.to_thread(embedder.encode, texts)

# ---------- Streaming Generator ----------
async def stream_questions(questions: List[str], chunk_size: int):
    for i in range(0, len(questions), chunk_size):
        yield questions[i:i+chunk_size]

# ---------- Main Pipeline ----------
async def process_questions(
    questions: List[str],
    batch_size: int,
    model_name: str,
    embedding_model: str,
    dsn: str,
    json_path: str
):
    embedder = SentenceTransformer(embedding_model)
    metadata = {
        "timestamp": datetime.utcnow().isoformat(),
        "batch_size": str(batch_size),
        "ollama_model": model_name,
        "embedding_model": embedding_model,
        "question_count": str(len(questions))
    }

    results = {}
    async for batch in stream_questions(questions, batch_size):
        logging.info(f"Processing batch of {len(batch)} questions...")
        try:
            answers = await fetch_batch(batch, model_name)
            embeddings = await async_embed(answers, embedder)
            for q, ans, emb in zip(batch, answers, embeddings):
                results[q] = {"answer": ans, "embedding": emb}
        except Exception as e:
            logging.error(f"Error processing batch: {e}")

    await save_to_postgres(results, dsn)
        export_to_json(results, json_path, metadata)
            logging.info("Processing complete.")

# ---------- PostgreSQL Insert ----------
async def save_to_postgres(results: Dict[str, Dict[str, object]], dsn: str):
    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cur:
                for question, data in results.items():
                    await cur.execute(
                        """
                        INSERT INTO qa_embeddings (question, answer, embedding)
                        VALUES (%s, %s, %s)
                        """,
                        (question, data["answer"], data["embedding"])
                    )
            await conn.commit()
        logging.info("Data saved to PostgreSQL.")
    except Exception as e:
        logging.error(f"Database error: {e}")

# ---------- JSON Export ----------
def export_to_json(results: Dict[str, Dict[str, object]], file_path: str, metadata: Dict[str, str]):
    try:
        serializable_results = {
            "metadata": metadata,
            "data": {
                q: {"answer": data["answer"], "embedding": list(data["embedding"])}
                for q, data in results.items()
            }
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        logging.info(f"Results exported to {file_path}")
    except Exception as e:
        logging.error(f"Error exporting JSON: {e}")

# ---------- Semantic Search ----------
async def semantic_search(query: str, embedder, dsn: str, top_k: int = 5):
    try:
        query_embedding = embedder.encode([query])[0]
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT question, answer
                    FROM qa_embeddings
                    ORDER BY embedding <-> %s
                    LIMIT %s
                    """,
                    (query_embedding, top_k)
                )
                rows = await cur.fetchall()
        return rows
    except Exception as e:
        logging.error(f"Semantic search error: {e}")
        return []

# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(description="Process questions with Ollama and store embeddings.")
    parser.add_argument("--dsn", required=True, help="PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)")
    parser.add_argument("--json", required=True, help="Path to export JSON file")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for Ollama requests")
    parser.add_argument("--model", default="llama2", help="Ollama model name")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    return parser.parse_args()

# ---------- Main ----------
async def main():
    args = parse_args()
    questions = [
        "What is the capital of France?",
        "Explain quantum entanglement in simple terms.",
        "What are the benefits of Python for data science?",
        "Who wrote 'The Art of War'?",
        "What is the speed of light?",
        "Define machine learning.",
        "What is the tallest mountain in the world?"
    ]

    await process_questions(
        questions,
        batch_size=args.batch_size,
        model_name=args.model,
        embedding_model=args.embedding_model,
        dsn=args.dsn,
        json_path=args.json
    )

    # Example semantic search
    embedder = SentenceTransformer(args.embedding_model)
    results = await semantic_search("What is quantum physics?", embedder, args.dsn)
    logging.info(f"Semantic search results: {results}")

if __name__ == "__main__":
    asyncio.run(main())



# python final_script.py \
#   --dsn "postgresql://user:password@localhost:5432/mydb" \
#   --json "qa_embeddings.json" \
#   --batch-size 5 \
#   --model "llama2" \
#   --embedding-model "all-MiniLM-L6-v2"
