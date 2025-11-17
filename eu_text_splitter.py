import re
import typing
from typing import List, Dict, Optional, Type, Any, Callable, Tuple, Generator, Any
import numpy as np
from itertools import product
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TextSplitter,
)
from langchain.schema import Document
import morfeusz2

from collections import Counter
import psycopg
from psycopg import sql
import traceback
import os
import src.utils.logger as logger_utils
from src.utils.dbsql import read_sql_files_for_table
from src.utils.select import select_id_informacji
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

def db_select_interptacje(conn_str: str, table_name: str):
    try:
        schema, table = table_name.split('.')
        with psycopg.connect(conn_str) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                query_sql = sql.SQL(""" SELECT
                id AS interpretacje_id,
                id_informacji,
                status_informacji,
                teza,
                dt_wyd,
                syg TEXT,
                tresc_interesariusz,
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
                return rows
    except Exception as e:
        logger.error("Error selecting top 1:")
        logger.error(traceback.format_exc())


#########################################################
def cut_text_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text.lower()))
        idx = match.start()
    except:
        idx = len(text)
    return text[0:idx]

def cut_text_by_regex(text: str):
    stems = ['poucze','informacja zakres',
                      'dodatkowe inform','informacja o zakres',
                      'uzasadnienie interpr',
                      'postępowanie przed sądami administ',
                      'zażalenie na postan',
                      'podstawa prawna','stronie przysługuje prawo','ocena stanowi']

    regex_parts = []
    for stem in stems:
        regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')

    for reg in regex_parts:
        print(f'Before cut_text_end: {len(text)} for "{reg}"')
        text = cut_text_word(text=text, pattern=reg)
        print(f'After cut_text_end: {len(text)}')
    return text

def index_text_start_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text))
        idx = match.start()
    except:
        idx = 0
    return idx

def cut_text_start_by_regex(text: str):
    stems = ["opis stanu faktyczn", "opisu stanu faktyczn", "opisowi stanu faktyczn",
        "opis stanu faktyczn", "opisem stanu faktyczn", "opiśie stanu faktyczn",
        "opiśie stanu faktyczn","opis zdarzenia przysz", 
        "opisu zdarzenia przysz", "opisowi zdarzenia przysz",
        "opis zdarzenia przysz", "opisem zdarzenia przysz", "opiśie zdarzenia przysz",
        "opiśie zdarzenia przysz","pytan"]

    regex_parts = []
    for stem in stems:
        regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')

    start_index = []
    for reg in regex_parts:
        #print(f'Before cut_text_start: {len(text)} for "{reg}"')
        idx = index_text_start_word(text=text.lower(), pattern=reg)
        start_index.append(idx)
    #print(f'start_index: {start_index}')


    start_index = min([i for i in start_index if i>0])
    if start_index is None or start_index == len(text):
        start_index=0
    return text[start_index:]

###################################################################


def split_text_by_header_regex(text: str) -> str:
    """
    Split text into chunk using header
    """
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]

    HEADER_PATTERN = r'([A-ZŚĆŻŹŃŁÓĘĄ][^\n]*[^\.\?!:;]\s*)$'
    chunks_content = []
    current_chunk = []
    is_initial_header_gathering = True

    for i, line in enumerate(text_lines):
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
                chunks_content.append(' '.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)


    if current_chunk:
        chunks_content.append(' '.join(current_chunk))

    return ' \n\n '.join(chunks_content)


def reconstruct_text(text: str, pattern: str, custom_separator: str) -> str:
    """
    Reconstruct text
    """
    # Reconstruct text with separator
    parts = re.split(pattern, text)
    reconstructed_text = ""
    for i in range(0, len(parts) - 1, 2):
        sentence = parts[i]
        whitespace = parts[i + 1]
        reconstructed_text += sentence + whitespace + custom_separator
    # Add the last sentence if it exists
    if len(parts) % 2 != 0:
        reconstructed_text += parts[-1]

    # Remove trailing separator if needed
    text = reconstructed_text.strip(custom_separator)
    return text

def remove_custom_sequence(text: str, separator: str) -> str:
    """
    Removes sequences like sep +  + sep from the text.
    """
    escaped_sep = re.escape(separator)
    pattern = rf"{escaped_sep}[\s\d\.:;?!,]*{escaped_sep}"
    cleaned_text = re.sub(pattern, separator, text)
    return cleaned_text

def experiment_text_splitting(
    text: str,
    chunk_sizes: List[int],
    chunk_overlaps: List[int],
    splitter_class: Type[TextSplitter] = RecursiveCharacterTextSplitter,
) -> Dict[str, List[str]]:
    """
    Splits text into chunks using different parameters and allows filtering.
    """
    results = {}
    logger.info(f"Starting Splitting Experiments on {splitter_class.__name__}")
    chunk_sizes_overlaps = list(product(chunk_sizes,chunk_overlaps, repeat=1))

    for size_overlap in chunk_sizes_overlaps:

        size, overlap = size_overlap
        if overlap >= size:
            logger.info(f"Skipping experiment: overlap ({overlap}) >= size ({size})")
            continue

        experiment_key = f"size_{size}_overlap_{overlap}"
        logger.info(f"\n=> Running experiment: {experiment_key}")


        # Initialize the specified text splitter
        try:
            text_splitter = splitter_class(
                chunk_size=size, chunk_overlap=overlap, separators = ['\n\n']
            )
        except Exception as e:
            logger.info(f"Error initializing splitter for {experiment_key}: {e}")
            continue

        # Split the text into chunks
        all_chunks = text_splitter.split_text(text)
        all_chunks = [chunk for chunk in all_chunks if len(chunk.strip()) > 50]
        logger.info(f"Total chunks created: {len(all_chunks)}")
        results[experiment_key] = all_chunks

        logger.info(
            f"Chunks after: {len(all_chunks)}"
        )

    return results


from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM


def test_plumm1():

    template = """Question: {question}

    Answer: Pomyslę i odpowiem po polsku."""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="PRIHLOP/PLLuM:12b", temperature=0.9, max_tokens=1024)

    chain = prompt | model

    response = chain.invoke({"question": "Jakie są rodzaje podatków w Polsce?"})
    print(response)





if __name__ == "__main__":


    rows = db_select_interptacje(conn_str=POSTGRES_DSN, table_name=TBL_INTERP)
    for row in rows[0:1]:
        # User choice
        id_informacji = row.get('id_informacji', [])
        tax_type = row.get('typ_podatku', [])
        teza_text = row.get('teza', [])
        interp_text = row.get('tresc_interesariusz', [])

        interp_text = cut_text_start_by_regex(interp_text)
        interp_text = cut_text_by_regex(interp_text)

        template = """Jestem ekspertem skarbowym. Otrzymałem treść wniosku: {question}

        Odpowiedz na pytania:
        1) jakie są słowa kluczowe
        2) stwórz 5 pytań do wypowiedzi
        Wyniki zwróć w postaci listy."""
        prompt = ChatPromptTemplate.from_template(template)

        model = OllamaLLM(model="hf.co/BMarcin/Llama-PLLuM-70B-chat-GGUF:Q4_K_M", temperature=0.9, max_tokens=1024)
        chain = prompt | model

        response = chain.invoke({"question": interp_text})
        print(response)

        # interp_text = split_text_by_header_regex(text=interp_text)

#hf.co/second-state/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0
        # print("\n" + "=" * 50)
        # print("### EXPERIMENT  ###")
        # print("=" * 50)
        # results = experiment_text_splitting(
        #     text=interp_text,
        #     chunk_sizes=[1000],
        #     chunk_overlaps=[200],
        # )
        # print(results)

