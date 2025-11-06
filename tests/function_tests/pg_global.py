
import os
import re
import time
import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from itertools import islice
import traceback
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

logger = logger_utils.setup_logger(name='pg_global')

def generate_create_table_sql(table_name: str, schema_dict: dict) -> str:
    columns = []
    for key, value in schema_dict.items():
        # Use custom type if provided, otherwise infer
        column_def = f"{key} {value}"
        columns.append(column_def)
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
    return create_table_sql

def create_table_db(connection_string:str, sql:str):
    try:
        with psycopg.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Psycopg3 connection test successful.")
    except Exception as e:
        logger.error(f"Error in :{create_table_db.__name__}")
        logger.error(traceback.format_exc())


def create_table_db_cursor_file(cursor, sql_file_path: str, db_schema_table: str ):
    try:
        with open(sql_file_path, "r") as file:
            sql_template = file.read()
        sql_query = sql_template.format(db_schema_table=db_schema_table)
        cursor.execute(sql_query)
        logger.info(f"Table {db_schema_table} created successfully.")
    except Exception as e:
        logger.error(f"Error creating table {db_schema_table}: {e}")


def generate_insert_sql(table_name, column_names):
    # Create a comma-separated string of column names
    if 'id' in column_names:
        column_names = column_names[1:]

    columns = ", ".join(column_names)
    # Create a string of placeholders for values
    placeholders = ", ".join(["%s"] * len(column_names))
    # Construct the SQL statement
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders});"
    return sql

def generate_update_sql(table_name, column_names, where_clause="id = %s"):
    # Create the SET part of the SQL statement
    set_clause = ", ".join([f"{col} = %s" for col in column_names])
    # Construct the full SQL statement
    sql = f"UPDATE {table_name} SET {set_clause} WHERE {where_clause};"
    return sql

# TODO connection string, insert_sql, columns by params in function 
# Insert chunk with error handling ---
# def insert_chunk_db(chunk):
#     try:
#         with psycopg.connect(POSTGRES_DSN) as conn:
#             with conn.cursor() as cur:
#                 # cur.execute("SELECT version();")
#                 # record = cur.fetchone()
#                 # print(chunk)
#                 # print([tuple(item.get(col, None) for col in COLUMNS) for item in chunk])
#                 cur.executemany(INSERT_SQL, [
#                     tuple(item.get(col, None) for col in COLUMNS)
#                     for item in chunk
#                 ])
#             conn.commit()
#         logger.info(f"Inserted chunk of {len(chunk)} records.")
#     except Exception as e:
#         logger.error("Error in insert_chunk:")
#         logger.error(traceback.format_exc())

# Chunking helper
def chunked_iterable(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def process_insert_chunk(data_list):
    chunk_size = 500
    chunks = list(chunked_iterable(data_list, chunk_size))
    try:
        with Pool(processes=cpu_count()) as pool:
            pool.map(insert_chunk, chunks)
        logger.info("All chunks inserted.")
    except Exception as e:
        logger.error("Error during multiprocessing insert:")
        logger.error(traceback.format_exc())


def select_table_db(connection_string:str, table_name:str):
    try:
        with psycopg.connect(connection_string) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 10")
                rows = cur.fetchall()
                for row in rows:
                    logger.info(f"Row: {row}")
    except Exception as e:
        logger.error("Error selecting top 10:")
        logger.error(traceback.format_exc())


def data_list_dict(data_list):
    try:
        with psycopg.connect(**DB_PARAMS) as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
                conn.commit()
        logger.info("Table created or already exists.")
    except Exception as e:
        logger.error("Error creating table:")
        logger.error(traceback.format_exc())
        return

    chunk_size = 100
    chunks = list(chunked_iterable(data_list, chunk_size))

    try:
        with Pool(processes=cpu_count()) as pool:
            pool.map(insert_chunk, chunks)
        logger.info("All chunks inserted.")
    except Exception as e:
        logger.error("Error during multiprocessing insert:")
        logger.error(traceback.format_exc())

    try:
        with psycopg.connect(**DB_PARAMS) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 10")
                rows = cur.fetchall()
                for row in rows:
                    logger.info(f"Row: {row}")
    except Exception as e:
        logger.error("Error selecting top 10:")
        logger.error(traceback.format_exc())

#     data_list = [
#         {'id_informacji': '1', 'kategoria_informacji': 'Podatek', 'status_informacji': 'Opublikowana', 'data_publikacji': '2025-10-13'},
#         # Add more dictionaries...
#     ]
#     main(data_list)


def create_tabe_db_jsonb():
    import psycopg
    with psycopg.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS eu_data (
                    id SERIAL PRIMARY KEY,
                    data JSONB
                );
                """
            )
        conn.commit()


def generate_sql_files_for_table(table_name: str, schema_dict: dict, output_dir: str):
    """
    Generates and saves CREATE TABLE and INSERT INTO SQL files for a given table.

    Parameters:
    - table_name (str): Full name of the table (e.g., 'public.interpretacje').
    - schema_dict (dict): Dictionary representing the table schema.
    - output_dir (str): Directory path where SQL files will be saved.
    """
    # Generate CREATE TABLE SQL
    create_sql = generate_create_table_sql(table_name=table_name, schema_dict=schema_dict)
    create_sql_path = f'{output_dir}create_{table_name}.sql'
    with open(create_sql_path, 'w') as file:
        file.write(create_sql)

    # Generate INSERT INTO SQL
    insert_sql = generate_insert_sql(table_name=table_name, column_names=list(schema_dict))
    insert_sql_path = f'{output_dir}insert_{table_name}.sql'
    with open(insert_sql_path, 'w') as file:
        file.write(insert_sql)


def extract_columns_and_values(insert_sql):
    # Extract column names
    # columns_match = re.search(r"INSERT\s+INTO\s+\w+\s*\((.*?)\)", insert_sql, re.IGNORECASE)
    columns_match = re.search(r'INSERT\s+INTO\s+\S+\s*\((.*?)\)', sql_query, re.IGNORECASE)
    columns = [col.strip() for col in columns_match.group(1).split(',')] if columns_match else []

    # Extract values
    values_match = re.search(r"VALUES\s*\((.*?)\)", insert_sql, re.IGNORECASE)
    if values_match:
        # This handles quoted strings and numbers
        raw_values = values_match.group(1)
        # Split values while respecting quotes
        values = re.findall(r"(?:'[^']*'|[^,]+)", raw_values)
        values = [val.strip().strip("'") for val in values]
    else:
        values = []

    return dict(zip(columns, values))


def extract_column_names(sql_query:str):
    # Match the part between parentheses after the table name
    match = re.search(r'INSERT\s+INTO\s+\S+\s*\((.*?)\)', sql_query, re.IGNORECASE)
    if match:
        columns_str = match.group(1)
        columns = [col.strip() for col in columns_str.split(',')]
        return columns
    else:
        return []


# Example usage
# sql = "INSERT INTO employees (id, name, age, department) VALUES (1, 'Alice', 30, 'HR');"
# columns = extract_column_names(sql)
# print(columns)


def read_sql_files_for_table(table_name: str, input_dir: str) -> dict:
    """
    Reads CREATE TABLE and INSERT INTO SQL files for a given table.

    Parameters:
    - table_name (str): Full name of the table (e.g., 'public.interpretacje').
    - input_dir (str): Directory path where SQL files are located.

    Returns:
    - dict: A dictionary with keys 'create_sql' and 'insert_sql' containing the SQL strings.
    """
    create_sql_path = f'{input_dir}create_{table_name}.sql'
    insert_sql_path = f'{input_dir}insert_{table_name}.sql'

    with open(create_sql_path, 'r') as file:
        create_sql = file.read()

    with open(insert_sql_path, 'r') as file:
        insert_sql = file.read()

    return {
        'create_sql': create_sql,
        'insert_sql': insert_sql,
        'columns_sql': extract_column_names(sql_query=insert_sql)
    }


def extract_columns_and_values(insert_sql):
    # Extract column names
    # columns_match = re.search(r"INSERT\s+INTO\s+\w+\s*\((.*?)\)", insert_sql, re.IGNORECASE)
    columns_match = re.search(r'INSERT\s+INTO\s+\S+\s*\((.*?)\)', sql_query, re.IGNORECASE)
    columns = [col.strip() for col in columns_match.group(1).split(',')] if columns_match else []

    # Extract values
    values_match = re.search(r"VALUES\s*\((.*?)\)", insert_sql, re.IGNORECASE)
    if values_match:
        # This handles quoted strings and numbers
        raw_values = values_match.group(1)
        # Split values while respecting quotes
        values = re.findall(r"(?:'[^']*'|[^,]+)", raw_values)
        values = [val.strip().strip("'") for val in values]
    else:
        values = []

    return dict(zip(columns, values))

# Example usage
# sql = "INSERT INTO employees (id, name, age, department) VALUES (1, 'Alice', 30, 'HR');"
# result = extract_columns_and_values(sql)
