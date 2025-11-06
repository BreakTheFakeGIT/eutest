import psycopg
from multiprocessing import Pool, cpu_count
from itertools import islice
import logging
import traceback

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bulk_insert.log"),
        logging.StreamHandler()
    ]
)

# --- Database connection parameters ---
DB_PARAMS = {
    "dbname": "your_db",
    "user": "your_user",
    "password": "your_password",
    "host": "localhost",
    "port": "5432"
}

# --- Table and columns ---
TABLE_NAME = "informacje"
COLUMNS = [
    'id_informacji', 'kategoria_informacji', 'status_informacji', 'data_publikacji',
    'teza', 'dt_wyd', 'syg', 'slowa_kluczowe', 'przepisy', 'zagadnienia',
    'tresc_interesariusz', 'tresc_interesariusz_html', 'id_eu', 'kod_eu', 'opis_eu',
    'wartosc_eu', 'przepisy_opis_eu', 'przepisy_wartosc_eu', 'zagadnienia_opis_eu',
    'zagadnienia_wartosc_eu', 'slowa_kluczowe_opis_eu', 'slowa_kluczowe_wartosc_eu',
    'typ_podatku', 'typ_podatku_opis', 'versionid', 'nazwa', 'szablonid', 'wersjaszablonuid'
]

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    {', '.join([f"{col} TEXT" for col in COLUMNS])}
);
"""

PLACEHOLDERS = ", ".join(["%s"] * len(COLUMNS))
INSERT_SQL = f"INSERT INTO {TABLE_NAME} ({', '.join(COLUMNS)}) VALUES ({PLACEHOLDERS})"

# --- Insert chunk with error handling ---
def insert_chunk(chunk):
    try:
        with psycopg.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, [
                    tuple(item.get(col, None) for col in COLUMNS)
                    for item in chunk
                ])
                conn.commit()
        logging.info(f"Inserted chunk of {len(chunk)} records.")
    except Exception as e:
        logging.error("Error in insert_chunk:")
        logging.error(traceback.format_exc())

# --- Chunking helper ---
def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

# --- Main function ---
def main(data_list):
    try:
        with psycopg.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                conn.commit()
        logging.info("Table created or already exists.")
    except Exception as e:
        logging.error("Error creating table:")
        logging.error(traceback.format_exc())
        return

    chunk_size = 100
    chunks = list(chunked_iterable(data_list, chunk_size))

    try:
        with Pool(processes=cpu_count()) as pool:
            pool.map(insert_chunk, chunks)
        logging.info("All chunks inserted.")
    except Exception as e:
        logging.error("Error during multiprocessing insert:")
        logging.error(traceback.format_exc())

    try:
        with psycopg.connect(**DB_PARAMS) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 10")
                rows = cur.fetchall()
                for row in rows:
                    logging.info(f"Row: {row}")
    except Exception as e:
        logging.error("Error selecting top 10:")
        logging.error(traceback.format_exc())

# --- Example usage ---
if __name__ == "__main__":
    data_list = [
        {'id_informacji': '1', 'kategoria_informacji': 'Podatek', 'status_informacji': 'Opublikowana', 'data_publikacji': '2025-10-13'},
        # Add more dictionaries...
    ]
    main(data_list)