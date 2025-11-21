
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

def db_select_interptacje(conn_str: str, table_name: str):
    try:
        schema, _ = table_name.split('.')
        with psycopg.connect(conn_str) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query_sql = sql.SQL("""SELECT
                    id AS interpretacje_id,
                    id_informacji,
                    teza,
                    dt_wyd,
                    syg,
                    tresc_interesariusz,
                    przepisy_opis_eu,
                    przepisy_wartosc_eu,
                    slowa_kluczowe_wartosc_eu,
                    typ_podatku,
                    typ_podatku_opis,
                    dataczas_akt
                FROM {schema}.interpretacje AS ta
                WHERE 1=1
                    AND kategoria_informacji = 1
                    AND szablonid IN (1,2)
                    AND EXISTS(SELECT 1 FROM {schema}.interpretacje_podobne AS tb WHERE ta.id_informacji = tb.id_informacji_powiazane)
                ORDER BY id_informacji DESC
                LIMIT 10""").format(schema=sql.Identifier(schema))
                cur.execute(query_sql)
                rows = cur.fetchall()
                return rows
    except Exception as e:
        logger.error("Error selecting top 1:")
        logger.error(traceback.format_exc())


