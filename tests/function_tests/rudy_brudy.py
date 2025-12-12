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

def sql_interpetacje(table_name: str, limit: int) -> str:
    schema, table = table_name.split('.')
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
                    LIMIT {limit}""").format(schema=sql.Identifier(schema), table=sql.Identifier(table), limit=sql.Identifier(limit))

    return query_sql



def fetch_in_batches(
    conn_str: str,
    query: str,
    batch_size: int = 1000
) -> Generator[list[Any], None, None]:
    """
    """
    # Connect to the database
    with psycopg.connect(conn_str) as conn:
        # Create a server-side cursor (named cursor)
        with conn.cursor(name="batch_cursor") as cur:
            cur.execute(query)
            
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                yield batch



def select_data_interp(connection_string: str, table_name: str):
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
                LIMIT 13""").format(schema=sql.Identifier(schema), table=sql.Identifier(table))
                cur.execute(query_sql)
                rows = cur.fetchall()
                # for row in rows:
                #     logger.info(f"Row: {row}")
                return rows
    except Exception as e:
        logger.error("Error selecting top 1:")
        logger.error(traceback.format_exc())



def cut_text_start_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text))
        idx = match.start()+len(pattern)
    except:
        idx = 0

    return text[idx:]

def cut_text_end_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text))
        idx = match.start()
    except:
        idx = len(text)

    return text[0:idx]


def sorted_bigram_counts(text: str):
    words = text.strip().split()

    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    sorted_bigrams = [(pair[0], pair[1], count) for pair, count in bigram_counts.most_common()]

    return sorted_bigrams

def filter_bigrams_word(bigram_list, word):
    return [bigram for bigram in bigram_list if word in bigram[:2]]


def filter_bigrams_regex(bigram_list, patterns):
    compiled_patterns = [re.compile(p) for p in patterns]
    filtered = []
    for bigram in bigram_list:
        word1, word2, count = bigram
        if any(p.search(word1) or p.search(word2) for p in compiled_patterns):
            filtered.append(bigram)
    
    return filtered

# # Example usage:
# text = "abc cd ef abc cd cd ef abcd ef"
# bigrams = sorted_bigram_counts(text)
# print("All bigrams:", bigrams)

# # Filter bigrams using regex patterns
# patterns = [r"^abc", r"ef$"]  # Matches words starting with 'abc' or ending with 'ef'
# filtered = filter_bigrams_regex(bigrams, patterns)
# print("Filtered bigrams:", filtered)



















def process_polish_text2(text: str) -> List[Document]:#Tuple[str, List[Document]]:

    # Definiujemy rdzenie (stemy), na podstawie których będziemy dzielić.
    # To czyni skrypt odpornym na odmianę przez przypadki (inflekcję).
    stems_to_split = ['ocen', 'stan faktyczn', 'pytan','opis']
    stem_to_remove = ['poucze','uzasad','funkc','stanowisk']

    # Tworzymy jeden wzorzec regex do znalezienia *wszystkich* fraz kluczowych
    all_stems = stems_to_split + stem_to_remove

    # \b - granica słowa (word boundary), aby nie dopasować "dopouczenie"
    # \w* - dopasowuje resztę słowa (czyli odmianę)
    # .replace(' ', r'\s+') - obsługuje frazy wielowyrazowe jak "stan faktyczny"
    regex_parts = []
    for stem in all_stems:
        regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')
        print(regex_parts)

    # Łączymy wszystko w jeden wzorzec.
    # Używamy JEDNEJ grupy przechwytującej (nawiasy zewnętrzne),
    # aby re.split() zachował separatory (nasze frazy).
    pattern_str = r'(' + r'|'.join(regex_parts) + r')'

    # Kompilujemy regex, ignorując wielkość liter (re.IGNORECASE)
    pattern = re.compile(pattern_str, flags=re.IGNORECASE)

    # Dzielimy tekst. Wynik będzie listą:
    # ['tekst przed', 'separator1', 'tekst po 1', 'separator2', 'tekst po 2', ...]
    chunks = pattern.split(text)



def regex_pattern(stems: List[str]):
    regex_parts = []
    for stem in stems:
        regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')

    pattern_str = r'(' + r'|'.join(regex_parts) + r')'
    print(pattern_str)
    pattern = re.compile(pattern_str, flags=re.IGNORECASE)
    print(pattern)
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


def cut_text_end_list_pattern(text: str, text_end: list[str]) -> str:
    """ Function """
    for pat in text_end:
        print(f'Before: {len(text)} for "{pat}"')
        text = cut_text_end_word(text=text, pattern = pat)
        print(f'After: {len(text)}')
    return text


def cut_text_start_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text))
        idx = match.start()+len(pattern)
    except:
        idx = 0

    return text[idx:]

def cut_text_end_word(text: str, pattern: str) -> str:
    """ Function """
    try:
        match=(re.search(pattern, text))
        idx = match.start()
    except:
        idx = len(text)

    return text[0:idx]







def split_text(text: str) -> str:
    """
    Split text into chunk using header
    """

    cut_text_end_word(text: str, pattern: str)


    # stems_to_split = ['ocen', 'stan faktyczn', 'pytan','opis']
    # stem_to_remove = ['poucze','uzasad','funkc','stanowisk']
    # all_stems = stems_to_split + stem_to_remove

    # regex_parts = []
    # for stem in all_stems:
    #     regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')
    #     print(regex_parts)


    # pattern_str = r'(' + r'|'.join(regex_parts) + r')'
    # pattern = re.compile(pattern_str, flags=re.IGNORECASE)
    # print(pattern)
    # chunks = pattern.split(text)



    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    print(text_lines)
    # stems_to_split = ['ocen', 'stan faktyczn', 'pytan','opis zdarz','treść']
    # stem_to_remove = ['poucze','uzasad','funkcj','dodatkowe info','informacja o zakresie rozstrzyg',
    # 'zakres wniosk', 'ocena stanowiska','prawidłowe','szanowni państ','w sprawie oceny skutków']
    # all_stems = stems_to_split + stem_to_remove



    # start_index, last_excluded_index = filter_sentences_with_regex(sentences=text_lines, target_patterns=stems_to_split)
    # print(f'Filter_sentences_with_regex: {start_index} - {last_excluded_index}.')
    # text_lines = text_lines[start_index:last_excluded_index]

    # pattern = regex_pattern(stems = stem_to_remove)
    # print(pattern)



# This block demonstrates how to use the function.
if __name__ == "__main__":

    rows = select_data_interp(connection_string=POSTGRES_DSN, table_name=TBL_INTERP)
    for row in rows[0:1]:
        # User choice
        tax_type = row.get('typ_podatku', [])
        teza_text = row.get('teza', [])
        interp_text = row.get('tresc_interesariusz', [])


    # bigram_list = sorted_bigram_counts(text=interp_text)
    # patterns = [r"ocena"] 
    # filtered = filter_bigrams_regex(bigram_list=bigram_list, patterns=patterns)
    
    # print("Filtered bigrams:", filtered)


    #chunks = split_text_by_header_regex(text=interp_text)
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