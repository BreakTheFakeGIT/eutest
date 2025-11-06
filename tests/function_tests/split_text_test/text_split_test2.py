import re
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
import psycopg

def custom_text_splitter_and_insert(
    text: str,
    chunk_sizes: List[int],
    overlaps: List[int],
    exclude_phrases: Optional[List[str]] = None,
    exclude_words: Optional[List[str]] = None,
    exclude_regex: Optional[List[str]] = None,
    split_methods: Optional[List[str]] = None,
    db_config: dict = None,  # e.g., {"dbname": "yourdb", "user": "youruser", "password": "yourpass", "host": "localhost", "port": 5432},
    table_name: str = "text_chunks"
):
    # Apply exclusions
    if exclude_phrases:
        for phrase in exclude_phrases:
            text = text.replace(phrase, "")
    if exclude_words:
        for word in exclude_words:
            text = re.sub(rf"\b{re.escape(word)}\b", "", text)
    if exclude_regex:
        for pattern in exclude_regex:
            text = re.sub(pattern, "", text)

    # Connect to PostgreSQL
    with psycopg.connect(**db_config) as conn:
        with conn.cursor() as cur:
            # Ensure table exists
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    method TEXT,
                    chunk_size INTEGER,
                    overlap INTEGER,
                    chunk TEXT
                );
            """)

            # Loop through chunk sizes and overlaps
            for method in split_methods or ["recursive"]:
                for chunk_size in chunk_sizes:
                    for overlap in overlaps:
                        if method == "recursive":
                            splitter = RecursiveCharacterTextSplitter(
                                chunk_size=chunk_size,
                                chunk_overlap=overlap
                            )
                            chunks = splitter.split_text(text)
                        elif method == "sentence":
                            sentences = re.split(r'(?<=[.!?]) +', text)
                            chunks = []
                            for i in range(0, len(sentences), chunk_size):
                                chunk = " ".join(sentences[i:i + chunk_size])
                                chunks.append(chunk)
                        else:
                            raise ValueError(f"Unknown split method: {method}")

                        # Insert chunks into DB
                        for chunk in chunks:
                            cur.execute(f"""
                                INSERT INTO {table_name} (method, chunk_size, overlap, chunk)
                                VALUES (%s, %s, %s, %s);
                            """, (method, chunk_size, overlap, chunk))

        conn.commit()

text = "This is a sample text. It contains multiple sentences. Some words should be excluded. Let's test the splitter."

db_config = {
    "dbname": "yourdb",
    "user": "youruser",
    "password": "yourpass",
    "host": "localhost",
    "port": 5432
}

custom_text_splitter_and_insert(
    text=text,
    chunk_sizes=[50, 100],
    overlaps=[10, 20],
    exclude_phrases=["excluded"],
    exclude_words=["sample"],
    exclude_regex=[r"\btest\b"],
    split_methods=["recursive", "sentence"],
    db_config=db_config,
    table_name="text_chunks"
)


import psycopg
from typing import List, Optional

def fetch_chunks_from_db(
    db_config: dict,
    table_name: str = "text_chunks",
    method: Optional[str] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    limit: Optional[int] = None
) -> List[str]:
    query = f"SELECT chunk FROM {table_name} WHERE TRUE"
    params = []

    if method:
        query += " AND method = %s"
        params.append(method)
    if chunk_size:
        query += " AND chunk_size = %s"
        params.append(chunk_size)
    if overlap:
        query += " AND overlap = %s"
        params.append(overlap)
    if limit:
        query += " LIMIT %s"
        params.append(limit)

    chunks = []
    with psycopg.connect(**db_config) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
            chunks = [row[0] for row in rows]

    return chunks



db_config = {
    "dbname": "yourdb",
    "user": "youruser",
    "password": "yourpass",
    "host": "localhost",
    "port": 5432
}

chunks = fetch_chunks_from_db(
    db_config=db_config,
    method="recursive",
    chunk_size=100,
    overlap=20,
    limit=10
)

for i, chunk in enumerate(chunks, 1):
    print(f"Chunk {i}:\n{chunk}\n")