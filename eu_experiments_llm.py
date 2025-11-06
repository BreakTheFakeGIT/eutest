import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from psycopg import connect
from psycopg.rows import dict_row

###################################
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")


def process_text_and_store(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    connection_string: str,
    table_name: str = "text_chunks",
):
    # 1. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    print(chunks)

    # 2. Create embeddings using local Ollama
    embedder = OllamaEmbeddings(model="llama2")  # Change model name if needed
    embeddings = embedder.embed_documents(chunks)
    print(len(embeddings))

    # # 3. Connect to PostgreSQL
    # with connect(connection_string, row_factory=dict_row) as conn:
    #     with conn.cursor() as cur:
    #         # Optional: Create table if not exists
    #         cur.execute(f"""
    #             CREATE TABLE IF NOT EXISTS {table_name} (
    #                 id SERIAL PRIMARY KEY,
    #                 chunk TEXT,
    #                 embedding vector[128]
    #             );
    #         """)
    #         # 4. Insert chunks and embeddings
    #         for chunk, embedding in zip(chunks, embeddings):
    #             cur.execute(
    #                 f"INSERT INTO {table_name} (chunk, embedding) VALUES (%s, %s);",
    #                 (chunk, embedding)
    #             )
    #         conn.commit()

    #print(f"Inserted {len(chunks)} chunks into {table_name}.")



# # === Test Setup ===
if __name__ == "__main__":
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables chaining together components like LLMs, prompts, memory, and agents.
    LangChain supports integrations with many tools including Ollama, FAISS, and PostgreSQL.
    """
    chunk_size = 100
    chunk_overlap = 20

    process_text_and_store(
        text=sample_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        connection_string = POSTGRES_DSN,
        table_name = "text_chunks",
    )

    # print("\n=== Results from Database ===")
    # for row in results:
    #     print(row)