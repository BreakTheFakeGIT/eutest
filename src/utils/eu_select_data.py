import re
import typing
from typing import List, Dict, Optional, Type, Any, Callable, Tuple
import numpy as np
from itertools import product
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TextSplitter,
)
from langchain.schema import Document
import morfeusz2

import psycopg
from psycopg import sql
import traceback
import os
import src.utils.logger as logger_utils
from src.utils.dbsql import read_sql_files_for_table
from dotenv import load_dotenv
load_dotenv()


###################################
TBL_INTERP = os.environ.get("TBL_INTERP")

TBL_TEXT_SPLIT = os.environ.get("TBL_TEXT_SPLIT")
EU_SQL = os.environ.get("EU_SQL")
READ_SQL = read_sql_files_for_table(table_name=TBL_TEXT_SPLIT, input_dir=EU_SQL)
INSERT_SQL = READ_SQL.get('insert_sql',[])
COLUMNS = READ_SQL.get('columns_sql',[])
POSTGRES_DSN = os.environ.get("POSTGRES_DSN")

###################################
logger = logger_utils.setup_logger(name=f'eu_select_data')

def select_data(conn_str: str, table_name: str):
    try:
        schema, table = table_name.split('.')
        with psycopg.connect(conn_str) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query_sql = sql.SQL(""" SELECT * FROM {schema}.{table} 
                                    LIMIT 10""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
                cur.execute(query_sql)
                rows = cur.fetchall()
                # for row in rows:
                #     logger.info(f"Row: {row}")
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
