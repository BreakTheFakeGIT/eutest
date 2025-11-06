import os
import re
import time
import psycopg
import traceback
from multiprocessing import Pool, cpu_count
from src.utils.extract_json import single_json_line
from src.utils.dbsql import read_sql_files_for_table, chunked_iterable
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

###################################
TABLE_NAME = os.environ.get("TBL_INTERP")
EU_DICT = os.environ.get("EU_DICT")
EU_SQL = os.environ.get("EU_SQL")
READ_SQL = read_sql_files_for_table(table_name=TABLE_NAME, input_dir=EU_SQL)
INSERT_SQL = READ_SQL.get('insert_sql',[])
COLUMNS = READ_SQL.get('columns_sql',[])
CONSOLIDATE_JSON_FOLDER = os.environ.get("CONSOLIDATE_JSON_FOLDER")
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################
logger = logger_utils.setup_logger(name=f'eu_load_json_{TABLE_NAME}')
logger.info(f"TABLE_NAME path: {TABLE_NAME}")
logger.info(f"EU_DICT path: {EU_DICT}")
logger.info(f"EU_SQL path: {EU_SQL}")
logger.info(f"COLUMNS path: {COLUMNS}")
logger.info(f'CONSOLIDATE_JSON_FOLDER path: {CONSOLIDATE_JSON_FOLDER}')

###################################
dir_list = os.listdir(CONSOLIDATE_JSON_FOLDER)
dir_list.sort(reverse=False, key=lambda x: int(x.split('_')[-1].split('.')[0]))
logger.info(f'CONSOLIDATE_JSON list: {dir_list}')

###################################

def insert_chunk_db(chunk):
    """Insert chunk with error handling"""
    try:
        with psycopg.connect(POSTGRES_DSN) as conn:
            with conn.cursor() as cur:
                cur.executemany(INSERT_SQL, [
                    tuple(item.get(col, None) for col in COLUMNS)
                    for item in chunk
                ])
            conn.commit()
        logger.info(f"Inserted chunk of {len(chunk)} records.")
    except Exception as e:
        logger.error("Error in insert_chunk:")
        logger.error(traceback.format_exc())


def process_insert_chunk(data_list):
    chunk_size = 1000
    chunks = list(chunked_iterable(data_list, chunk_size))
    try:
        with Pool(processes=cpu_count()) as pool:
            pool.map(insert_chunk_db, chunks)
        logger.info("All chunks inserted.")
    except Exception as e:
        logger.error("Error during multiprocessing insert:")
        logger.error(traceback.format_exc())

def process_json_file(file_path):
    """ Process the entire JSON file using multiprocessing """
    logger.info(f'Start processing file: {file_path}')
    with open(file_path, 'r', encoding="utf8", errors="ignore") as f:
        lines = f.readlines()
    logger.info(f'Number of lines to process: {len(lines)}')

    with Pool(processes=max(cpu_count(), 8)) as pool:
        results = pool.map(single_json_line, lines)
    return results

def main_process():
    """ Process M """
    for file in dir_list:
        file_path = os.path.join(CONSOLIDATE_JSON_FOLDER, file)
        if not file.startswith('consolidate_json_') or not file.endswith('.json'):
            logger.info(f'Skipping file: {file}')
            continue

        t0 = time.time()
        try: 
            results = process_json_file(file_path)
            process_insert_chunk(data_list=results)
            # for result in results[0:1]:
            #     logger.info(f"Results process_json_file for verification JSON object: {result}")
            #     logger.info(f'Tax: {result.get('typ_podatku', [])}')
        except Exception as e:
            logger.error(f"Error file {file}")
            logger.error(traceback.format_exc())
            continue

        t1 = time.time() - t0
        logger.info(f'End processing file.\nTime: {t1:.2f} seconds.\nNumber of processed JSON objects: {len(results)}.\nAvg: {len(results)/t1:.2f} seconds.\nCPU count: {cpu_count()}. File: {file}.')
        del results

        try:
            with psycopg.connect(POSTGRES_DSN) as conn:
                with conn.pipeline() as p, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute(f"SELECT id_informacji, typ_podatku, slowa_kluczowe_wartosc_eu FROM {TABLE_NAME} LIMIT 1")
                    rows = cur.fetchall()
                    for row in rows:
                        logger.info(f"Row: {row}")
        except Exception as e:
            logger.error("Error selecting top 1:")
            logger.error(traceback.format_exc())

        t2 = time.time() - t0
        logger.info(f'Total time: {t2:.2f} seconds')

    with psycopg.connect(POSTGRES_DSN) as conn:
        with conn.pipeline() as p, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(f"SELECT count(*) FROM {TABLE_NAME}")
            row1 = cur.fetchone().get('count',0)
            cur.execute(f"DELETE FROM {TABLE_NAME} ta \
            USING {TABLE_NAME} tb \
            WHERE ta.id_informacji = tb.id_informacji \
            AND ta.id < tb.id;"
            )
            cur.execute(f"SELECT count(*) FROM {TABLE_NAME}")
            row2 = cur.fetchone().get('count',0)
            logger.info(f"DELETE FROM {TABLE_NAME} WHERE id NOT IN ...\nNumber of deleted rows BEFORE: {row1}, AFTER: {row2} and DIFFERENCE: {row2-row1}.")
            p.sync()

###################################
if __name__ == "__main__":


    # Filter the list using the pattern
    pattern = re.compile(r'consolidate_json_([1-9]\d*|[0-9]\d{2,})\.json')
    dir_list = [filename for filename in dir_list if pattern.match(filename)]

    # Find the index of the starting file
    start_file = 'consolidate_json_11.json'
    start_index = dir_list.index(start_file)
    dir_list = dir_list[start_index:]

    main_process()
