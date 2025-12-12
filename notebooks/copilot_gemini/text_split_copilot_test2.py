import psycopg
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import initialize_agent, AgentType
from langchain.llms import Ollama
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

import os
from dotenv import load_dotenv
load_dotenv()
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")
db_config = {
    'llm_model': 'hf.co/mradermacher/Llama-PLLuM-70B-instruct-GGUF:Q5_K_M'
    'embedding_model': "llama3.2:latest"
}


def process_text_and_store(text, chunk_size, chunk_overlap, prompts, db_config,llm_model, embedding_model):
    # 1. Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)

    # 2. Create embeddings
    embedding_model = OllamaEmbeddings(model="llama3.2:latest")  # Replace with your local model name
    embeddings = embedding_model.embed_documents(chunks)

    # 3. Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)

    # 4. Initialize agent
    llm = Ollama(model="hf.co/mradermacher/Llama-PLLuM-70B-instruct-GGUF:Q5_K_M")  # Replace with your local model name
    agent = initialize_agent([], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

    # 5. Extract information using prompts
    extracted_data = []
    for i, chunk in enumerate(chunks):
        chunk_info = {"chunk_index": i, "chunk_text": chunk, "similarities": similarity_matrix[i].tolist()}
        for prompt in prompts:
            response = agent.run(f"{prompt}\n\n{chunk}")
            chunk_info[prompt] = response
        extracted_data.append(chunk_info)

    # 6. Insert into PostgreSQL using psycopg3
    with psycopg.connect(db_config) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk_analysis (
                    id SERIAL PRIMARY KEY,
                    chunk_index INT,
                    chunk_text TEXT,
                    similarities FLOAT8[],
                    prompt TEXT,
                    response TEXT
                )
            """)
            for data in extracted_data:
                for prompt in prompts:
                    cur.execute("""
                        INSERT INTO chunk_analysis (chunk_index, chunk_text, similarities, prompt, response)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (
                        data["chunk_index"],
                        data["chunk_text"],
                        data["similarities"],
                        prompt,
                        data[prompt]
                    ))
            conn.commit()

    # 7. Select and return data
    with psycopg.connect(**db_config) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM chunk_analysis")
            results = cur.fetchall()
            return results

# === Test Setup ===
if __name__ == "__main__":
    sample_text = """
    LangChain is a framework for developing applications powered by language models.
    It enables chaining together components like LLMs, prompts, memory, and agents.
    LangChain supports integrations with many tools including Ollama, FAISS, and PostgreSQL.
    """

    prompts = [
        "Summarize the content.",
        "List all tools mentioned."
    ]

    chunk_size = 100
    chunk_overlap = 20

    # db_config = {
    #     "host": "localhost",
    #     "port": 5432,
    #     "dbname": "langchain_test",
    #     "user": "postgres",
    #     "password": "your_password"  # Replace with your actual password
    # }

    results = process_text_and_store(
        text=sample_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        prompts=prompts,
        db_config=POSTGRES_DSN
    )

    print("\n=== Results from Database ===")
    for row in results:
        print(row)