
import asyncio
import ollama
import random
import json
import argparse
import logging
from typing import List, Dict, Union
from datetime import datetime
from sentence_transformers import SentenceTransformer
import psycopg
import numpy as np 
from torch.cuda import is_available

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
class AsyncEmbeddingManager:
    def __init__(self, model_names: List[str]):
        """
        Initialize multiple SentenceTransformer models.
        Automatically uses GPU if available.
        """
        self.device = "cuda" if is_available() else "cpu"
        self.models: Dict[str, SentenceTransformer] = {
            name: SentenceTransformer(name, device=self.device) for name in model_names
        }

    async def embed(self, model_name: str, text: Union[str, List[str]]) -> List[List[float]]:
        """ Async embedding for a single model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded. Available: {list(self.models.keys())}")

        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self.models[model_name].encode(normalize_embeddings=True, show_progress_bar=False), text)
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

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
            """
            Calculate cosine similarity between two vectors.
            Raises ValueError if vectors are empty or have different dimensions.
            """
            if not vec_a or not vec_b:
                raise ValueError("Vectors must not be empty.")

            a = np.array(vec_a)
            b = np.array(vec_b)

            if a.shape != b.shape:
                raise ValueError("Vectors must have the same dimensions.")

            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)

            if norm_a == 0 or norm_b == 0:
                raise ValueError("Vectors must not be zero vectors.")

            similarity = np.dot(a, b) / (norm_a * norm_b)
            return float(similarity)

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
    #json_path: str
):
    metadata = {
        #"timestamp": datetime.utcnow().isoformat(),
        "batch_size": str(batch_size),
        "ollama_model": model_name,
        "embedding_model": embedding_model,
        "question_count": str(len(questions))
    }
    async_emb_manager = AsyncEmbeddingManager(model_names=[embedding_model])
    results = {}
    async for batch in stream_questions(questions, batch_size):
        logging.info(f"Processing batch of {len(batch)} questions...")
        try:
            answers = await fetch_batch(batch, model_name)
            embeddings = await async_emb_manager.embed(model_name=embedding_model, text=answers) 
            for q, ans, emb in zip(batch, answers, embeddings):
                print(q)
                print(ans)
                print(emb)
                results[q] = {"answer": ans, "embedding": emb}
        except Exception as e:
            logging.error(f"Error processing batch: {e}")
    print(results)
    # Save to DB
    await save_to_postgres(results, dsn)

    # Export to JSON
    # export_to_json(results, json_path, metadata)

    logging.info("Processing complete.")



def to_pgvector_literal(vec: List[float]) -> str:
    """Convert a list of floats to pgvector textual literal: '[0.1,0.2,...]'"""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

# ---------- PostgreSQL Insert ----------
async def save_to_postgres(results: Dict[str, Dict[str, object]], dsn: str):
    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            async with conn.cursor() as cur:
                for question, data in results.items():
                    embedding = data["embedding"][0]
                    await cur.execute(
                        """
                        INSERT INTO qa_embeddings (question, answer, embedding)
                        VALUES (%s, %s, %s::vector)
                        """,
                        (question, data["answer"], embedding)
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
    #parser.add_argument("--dsn", required=True, help="PostgreSQL DSN (e.g., postgresql://user:pass@host:port/db)")
    #parser.add_argument("--json", required=True, help="Path to export JSON file")
    parser.add_argument("--batch-size", type=int, default=5, help="Batch size for Ollama requests")
    parser.add_argument("--model", default="hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0", help="Ollama model name")
    parser.add_argument("--embedding-model", default="/dane/models/sdadas/stella-pl-retrieval", help="SentenceTransformer model")
    return parser.parse_args()

# ---------- Main ----------
import os
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")
async def main():
    args = parse_args()
    questions = [
        "Jakie sa typy podatkow?",
        ""

    ]

    await process_questions(
        questions,
        batch_size=args.batch_size,
        model_name=args.model,
        embedding_model=args.embedding_model,
        dsn=POSTGRES_DSN,
        #json_path=args.json
    )

    # Example semantic search
    embedder = SentenceTransformer(args.embedding_model)
    results = await semantic_search("What is quantum physics?", embedder, dsn=POSTGRES_DSN)
    logging.info(f"Semantic search results: {results}")

asyncio.run(main())
