
import os
from typing import List
import psycopg
from psycopg import sql
import traceback
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

###################################
logger = logger_utils.setup_logger(name=f'select_data')

def select_id_informacji(conn_str: str, table_name: str) -> List[int]:
    try:
        schema, table = table_name.split('.')
        with psycopg.connect(conn_str) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query_sql = sql.SQL(""" SELECT DISTINCT id_informacji
                FROM {schema}.{table}
                ORDER BY id_informacji DESC""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
                cur.execute(query_sql)
                rows = cur.fetchall()
                # for row in rows:
                #     logger.info(f"Row: {row}")
                logger.info(f"Select rows for id_infromacji")
                rows = [row.get('id_informacji', []) for row in rows]
                return rows
    except Exception as e:
        logger.error("Error selecting top 1:")
        logger.error(traceback.format_exc())


