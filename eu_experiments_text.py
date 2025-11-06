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
logger = logger_utils.setup_logger(name=f'eu_experiments_text')

def select_data(connection_string: str, table_name: str):
    try:
        schema, table = table_name.split('.')
        with psycopg.connect(connection_string) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query_sql = sql.SQL(""" SELECT
                id AS interpretacje_id,
                id_informacji,
                kategoria_informacji,
                status_informacji,
                teza,
                dt_wyd,
                syg TEXT,
                tresc_interesariusz,
                tresc_interesariusz_html,
                przepisy_opis_eu,
                przepisy_wartosc_eu,
                slowa_kluczowe_wartosc_eu,
                typ_podatku,
                typ_podatku_opis,
                dataczas_akt
                FROM {schema}.{table}
                WHERE 1=1 
                    AND kategoria_informacji = 1
                    AND szablonid IN (1,2)
                ORDER BY id_informacji DESC
                LIMIT 10""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
                cur.execute(query_sql)
                rows = cur.fetchall()
                # for row in rows:
                #     logger.info(f"Row: {row}")
                return rows
    except Exception as e:
        logger.error("Error selecting top 1:")
        logger.error(traceback.format_exc())
    # finally:
    #     # Closing database connection.
    #     if conn:
    #         conn.close()
    #         logger.info("PostgreSQL connection is closed.")



def regex_pattern(stems: List[str]):
    regex_parts = []
    for stem in stems:
        regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')

    pattern_str = r'(' + r'|'.join(regex_parts) + r')'
    pattern = re.compile(pattern_str, flags=re.IGNORECASE)
    return pattern


def filter_sentences_with_regex(sentences, target_patterns):
    start_index = None
    last_excluded_index = None
    result = []

    # Compile regex patterns for efficiency and flexibility
    regexes = [re.compile(pattern, re.IGNORECASE) for pattern in target_patterns]

    # Step 1: Find the first index where any regex matches
    for i, sentence in enumerate(sentences):
        if any(regex.search(sentence) for regex in regexes):
            start_index = i
            break

    if start_index is None:
        return 0, len(sentences)

    # Step 2: Iterate from start_index and exclude sentences matching any regex
    for i in range(start_index, len(sentences)):
        sentence = sentences[i]
        if any(regex.search(sentence) for regex in regexes):
            last_excluded_index = i
            continue
        result.append(sentence)

    return start_index, last_excluded_index


def split_text_by_header_regex(text: str) -> List[Document]:
    """
    Split text into chunk using header
    """
    HEADER_PATTERN = r'([A-ZŚĆŻŹŃŁÓĘĄ][^\n]*[^\.\?!:;]\s*)$'
    chunks_content = []
    current_chunk = []
    is_initial_header_gathering = True
    stems_to_split = ['ocen', 'stan faktyczn', 'pytan','opis zdarz','treść']
    stem_to_remove = ['poucze','uzasad','funkcj','dodatkowe info','informacja o zakresie rozstrzyg',
    'zakres wniosk', 'ocena stanowiska','prawidłowe','szanowni państ','w sprawie oceny skutków']
    all_stems = stems_to_split + stem_to_remove


    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    start_index, last_excluded_index = filter_sentences_with_regex(sentences=text_lines, target_patterns=stems_to_split)
    print(f'Filter_sentences_with_regex: {start_index} - {last_excluded_index}.')
    text_lines = text_lines[start_index:last_excluded_index]

    pattern = regex_pattern(stems = stem_to_remove)
    print(pattern)

    #regexes_to_remove = [re.compile(pattern, re.IGNORECASE) in stem_to_remove]
    for i, line in enumerate(text_lines):

        # print(line)
        # if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in regexes_to_remove):
        #     print(line)
        #     break

        is_header = re.search(HEADER_PATTERN, line)
        if is_initial_header_gathering:
            if is_header:
                current_chunk.append(line)
            else:
                is_initial_header_gathering = False
                current_chunk.append(line)
            continue

        if is_header:
            if current_chunk:
                chunks_content.append('\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks_content.append('\n '.join(current_chunk))

    print(chunks_content)
    documents = [Document(page_content=content.strip()) for content in chunks_content if content.strip()]
    #documents = [content.strip() for content in chunks_content if content.strip()]
    return documents


# This block demonstrates how to use the function.
if __name__ == "__main__":

    rows = select_data(connection_string=POSTGRES_DSN, table_name=TBL_INTERP)
    for row in rows[0:1]:
        # User choice
        tax_type = row.get('typ_podatku', [])
        teza_text = row.get('teza', [])
        interp_text = row.get('tresc_interesariusz', [])


    chunks = split_text_by_header_regex(text=interp_text)
    # for i, chunk in enumerate(chunks):
    #     print(f"--- CHUNK {i+1} (Długość: {len(chunk.page_content)} znaków) ---")
    #     print(chunk.page_content)
    #     print("-" * 50)
    # print(f"Liczba uzyskanych fragmentów (chunks): {len(chunks)}\n")


# Opis zdarzenia przyszłego
# Pytanie
# Ocena stanowiska
# Dodatkowe informacje
# Pouczenie

