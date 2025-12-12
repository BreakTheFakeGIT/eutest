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

def filter_sentences_with_indices(sentences, target_words):
    start_index = None
    last_excluded_index = None
    result = []

    # Step 1: Find the first index where any target word appears
    for i, sentence in enumerate(sentences):
        if any(word in sentence.split() for word in target_words):
            start_index = i
            break

    if start_index is None:
        return [], None, None

    # Step 2: Iterate from start_index and exclude sentences with target words
    for i in range(start_index, len(sentences)):
        sentence = sentences[i]
        if any(word in sentence.split() for word in target_words):
            last_excluded_index = i
            continue
        result.append(sentence)

    return result, start_index, last_excluded_index

import re

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
        return [], None, None

    # Step 2: Iterate from start_index and exclude sentences matching any regex
    for i in range(start_index, len(sentences)):
        sentence = sentences[i]
        if any(regex.search(sentence) for regex in regexes):
            last_excluded_index = i
            continue
        result.append(sentence)

    return result, start_index, last_excluded_index



def process_polish_text2(text: str) -> List[Document]:#Tuple[str, List[Document]]:

    # Definiujemy rdzenie (stemy), na podstawie których będziemy dzielić.
    # To czyni skrypt odpornym na odmianę przez przypadki (inflekcję).
    stems_to_split = ['ocen', 'stan faktyczn', 'pytan','opis']
    stem_to_remove = ['poucze','uzasad','funkc','stanowisk']

    # Mapowanie rdzeni na kategorie (dla metadanych Langchain)
    stem_categories = {
        'ocen': 'assessment',
        'stan faktyczn': 'factual_status',
        'pytan': 'questions',
        'opis': 'description'
    }

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

    recombined_parts = []
    langchain_documents = []

    # Obsługa tekstu *przed* pierwszym znalezionym separatorem
    if chunks[0] and chunks[0].strip():
        initial_content = chunks[0].strip()
        recombined_parts.append(initial_content)
        langchain_documents.append(
            Document(
                page_content=initial_content,
                metadata={'source_phrase': 'initial_content'}
            )
        )

    # Przetwarzamy resztę listy parami (separator, treść)
    # Zaczynamy od indeksu 1, przeskakujemy co 2
    for i in range(1, len(chunks), 2):
        marker = chunks[i]

        if any(re.match(r'\b' + pattern, marker, flags=re.IGNORECASE) for pattern in stem_to_remove):
            continue


        # Jeśli nie jest do usunięcia, przetwarzamy go
        content = chunks[i+1].strip() if (i+1) < len(chunks) else ""

        if not content:
            continue # Ignorujemy puste sekcje

        # Identyfikujemy kategorię dla metadanych
        source_phrase = 'unknown'
        for stem, category in stem_categories.items():
            if re.match(r'\b' + stem.replace(' ', r'\s+'), marker, flags=re.IGNORECASE):
                source_phrase = category
                break

        # Łączymy marker (frazę) z jego treścią
        full_chunk_content = f"{marker.strip()}:\n{content}"

        # 1. Dodajemy do listy połączonych części
        recombined_parts.append(full_chunk_content)

        # 2. Tworzymy Dokument Langchain
        langchain_documents.append(
            Document(
                page_content=content, # Można też dać full_chunk_content
                metadata={'source_phrase': source_phrase, 'marker': marker.strip()}
            )
        )

    # Tworzymy finalny, połączony tekst, oddzielając sekcje podwójnym enterem
    final_recombined_string = "\n\n".join(recombined_parts)

    return langchain_documents




def process_polish_text(text: str) -> List[Document]:#Tuple[str, List[Document]]:

    stems_to_split = ['ocen', 'stan faktyczn', 'pytan','opis']
    stem_to_remove = ['poucze','uzasad','funkc','stanowisk']
    all_stems = stems_to_split + stem_to_remove

    regex_parts = []
    for stem in all_stems:
        regex_parts.append(r'\b' + stem.replace(' ', r'\s+') + r'\w*')
        print(regex_parts)


    pattern_str = r'(' + r'|'.join(regex_parts) + r')'
    pattern = re.compile(pattern_str, flags=re.IGNORECASE)
    print(pattern)
    chunks = pattern.split(text)

    recombined_parts = []
    if chunks[0] and chunks[0].strip():
        initial_content = chunks[0].strip()
        recombined_parts.append(initial_content)

    # Przetwarzamy resztę listy parami (separator, treść)

    for i in range(1, len(chunks), 2):
        marker = chunks[i]

        if any(re.match(r'\b' + pattern, marker, flags=re.IGNORECASE) for pattern in stem_to_remove):
            continue


        # Jeśli nie jest do usunięcia, przetwarzamy go
        content = chunks[i+1].strip() if (i+1) < len(chunks) else ""

        if not content:
            continue # Ignorujemy puste sekcje

        # Łączymy marker (frazę) z jego treścią
        full_chunk_content = f"{marker.strip()}:\n{content}"

        # 1. Dodajemy do listy połączonych części
        recombined_parts.append(full_chunk_content)


    # Tworzymy finalny, połączony tekst, oddzielając sekcje podwójnym enterem
    final_recombined_string = "\n\n".join(recombined_parts)

    return final_recombined_string














def split_text_by_header_regex(text: str) -> List[Document]:
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
                chunks_content.append('\n'.join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        else:
            current_chunk.append(line)


    if current_chunk:
        chunks_content.append('\n\n\n'.join(current_chunk))

    documents = [Document(page_content=content.strip()) for content in chunks_content if content.strip()]
    #documents = [content.strip() for content in chunks_content if content.strip()]
    return documents


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

# def remove_custom_sequence(text):
#     """ This pattern matches: |sep| followed by optional whitespace, digits, dots, colons, semicolons, then another |sep| """
#     pattern = r"\|sep\|[\s\d\.:;]*\|sep\|"
#     cleaned_text = re.sub(pattern, "", text)
#     return cleaned_text

def experiment_text_splitting(
    text: str,
    chunk_sizes: List[int],
    chunk_overlaps: List[int],
    splitter_class: Type[TextSplitter] = RecursiveCharacterTextSplitter,
    exclude_word: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    exclude_patterns: Optional[List[str]] = None,
    remove_separator: bool = True,
    **splitter_kwargs,
) -> Dict[str, List[str]]:
    """
    Splits text into chunks using different parameters and allows filtering.
    """




    list_patterns = [r'(?:\s*\n){2,}',
                r'\.\s{3,}',
                r'(?<=[:;!?])(\s{1,}+)',
                r'\n\n',
                r'\n\s*\n',
                r'\n\s*',
                r'\s*\n',
                r'\t\s*',
                r'\s*\n',
                r'(?<=[.!?:])(\s{3,}+)',
                ]

    for pattern in list_patterns:
        text = reconstruct_text(text=text, pattern=pattern, custom_separator=" |sep| ")
        text = remove_custom_sequence(text=text,separator=" |sep| ")
    logger.info(f"Reconstruct text with separator")


    results = {}
    regex_pattern = re.compile(exclude_regex) if exclude_regex else None
    word_pattern = (
        re.compile(r"\b" + re.escape(exclude_word) + r"\b", re.IGNORECASE)
        if exclude_word
        else None
    )

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
                chunk_size=size, chunk_overlap=overlap, **splitter_kwargs
            )
        except Exception as e:
            logger.info(f"Error initializing splitter for {experiment_key}: {e}")
            continue

        # Split the text into chunks
        all_chunks = text_splitter.split_text(text)


        # Filter the chunks based on exclusion criteria
        filtered_chunks = []
        for chunk in all_chunks:
            # Assume the chunk is valid until proven otherwise
            is_excluded = False

            # # Remove seperator
            # if remove_separator:
            #     chunk = chunk.replace(separator, " ").strip()

            # Check for whole word exclusion (case-insensitive)
            if not is_excluded and word_pattern and word_pattern.search(chunk):
                is_excluded = True
                logger.info(f"   - Excluding chunk (contains word '{exclude_word}')")

            # Check for regex exclusion
            if not is_excluded and regex_pattern and regex_pattern.search(chunk):
                is_excluded = True
                logger.info(f"   - Excluding chunk (matches regex '{exclude_regex}')")

            # Check for exclude patterns
            if not is_excluded and exclude_patterns and any(re.search(pattern, chunk, re.IGNORECASE) for pattern in exclude_patterns):
                is_excluded = True
                logger.info(f"   - Excluding chunk (exclude patterns '{','.join(exclude_patterns)}')")

    # if exclude_words:
    #     for word in exclude_words:
    #         text = re.sub(rf"\b{re.escape(word)}\b", "", text)
    # if exclude_regex:
    #     for pattern in exclude_regex:
    #         text = re.sub(pattern, "", text)

            if not is_excluded:
                filtered_chunks.append(chunk)

        results[experiment_key] = filtered_chunks

        logger.info(
            f"Original chunks created: {len(all_chunks)}, "
            f"Chunks after filtering: {len(filtered_chunks)}"
        )

    return results

def lemmatize(word):
    analyses = morf.analyse(word)
    if analyses:
        return analyses[0][2][1]  # lemma
    return word


# This block demonstrates how to use the function.
if __name__ == "__main__":

    rows = select_data(connection_string=POSTGRES_DSN, table_name=TBL_INTERP)
    for row in rows[0:1]:
        #logger.info(f"Row: {row}")
        #logger.info(f'Tax: {row.get('typ_podatku', [])}')

        # User choice
        tax_type = row.get('typ_podatku', [])
        teza_text = row.get('teza', [])
        interp_text = row.get('tresc_interesariusz', [])
        print(row.keys())


    # from  pystempel import Stemmer
    # stemmer = Stemmer.polimorf()

    # morf = morfeusz2.Morfeusz()
    # def lemmatize(word):
    #     analyses = morf.analyse(word)
    #     if analyses:
    #         return analyses[0][2][1]  # lemma
    #     return word

    # llm_output = ["stanowiska", "Dodatkowe", "informacje","jestem","interpretacji","indywidualnej"]
    # lem_out = [lemmatize(w) for w in llm_output]
    # print(lem_out)


    # for word in llm_output:
    #     print(stemmer(word))


    #print(f"Lematy dla '{word}':", lemmas)


    chunks = split_text_by_header_regex(text=interp_text)
    for i, chunk in enumerate(chunks):
        print(f"--- CHUNK {i+1} (Długość: {len(chunk.page_content)} znaków) ---")
        print(chunk.page_content)
        print("-" * 50)
    print(f"Liczba uzyskanych fragmentów (chunks): {len(chunks)}\n")
    print(process_polish_text(text=interp_text))



    #print(stems_to_split(text=interp_text))
    # chunks = process_polish_text(text=interp_text)
    # for i, chunk in enumerate(chunks):
    #     print(f"--- CHUNK {i+1} (Długość: {len(chunk.page_content)} znaków) ---")
    #     print(chunk.metadata)
    #     print(chunk.page_content)
        
    #     print("-" * 50)

    # # --- Experiment Parameters ---
    # chunk_sizes_to_test = [1500]
    # chunk_overlaps_to_test = [200]

    # print("\n" + "=" * 50)
    # print("### EXPERIMENT  ###")
    # print("=" * 50)
    # results = experiment_text_splitting(
    #     text=interp_text,
    #     chunk_sizes=chunk_sizes_to_test,
    #     chunk_overlaps=chunk_overlaps_to_test,
    #     exclude_regex = 'skarg',
    #     separators = ['|sep|']
    # )
    # print(results)
    # print([*results.keys()])
    # print(len(results.get([*results.keys()][0], [])))



# Opis zdarzenia przyszłego
# Pytanie
# Ocena stanowiska
# Dodatkowe informacje
# Pouczenie


















    # # --- 1. Basic Experiment: No exclusions ---
    # print("\n" + "=" * 50)
    # print("### EXPERIMENT 1: No Exclusions ###")
    # print("=" * 50)
    # results_no_exclusion = experiment_text_splitting(
    #     text=sample_text,
    #     chunk_sizes=chunk_sizes_to_test,
    #     chunk_overlaps=chunk_overlaps_to_test,
    # )
    # # Print one of the results for inspection
    # if results_no_exclusion:
    #     first_key = list(results_no_exclusion.keys())[0]
    #     print(f"\nSample result for '{first_key}':")
    #     for i, chunk in enumerate(results_no_exclusion[first_key][:2]): # Show first 2 chunks
    #          print(f"  Chunk {i+1}: '{chunk[:100]}...'")



    # # # Example of writing the results from the first experiment
    # # # You can change this to any other result set you want to save.
    # print("Attempting to write 'results_no_exclusion' to the database.")
    # print("NOTE: This will fail if you don't update connection details and have a running PostgreSQL server.")
    # print(results_different_splitter)
    # write_results_to_postgres(results_no_exclusion, connection_string=POSTGRES_DSN)



    # # --- 7. Retrieve Chunked Data from PostgreSQL ---
    # print("\n" + "=" * 50)
    # print("### EXPERIMENT 7: Retrieving Data from PostgreSQL ###")
    # print("=" * 50)

    # # Example of retrieving the data we just wrote for a specific experiment
    # retrieved_data = retrieve_chunks_from_postgres(
    #     connection_string=POSTGRES_DSN,
    #     experiment_key="size_150_overlap_30" # Should match a key from 'results_no_exclusion'
    # )

    # if retrieved_data:
    #     print(f"\nSuccessfully retrieved {len(retrieved_data)} chunks for 'size_150_overlap_30'.")
    #     print("First retrieved chunk:")
    #     # Print the first result pretty
    #     first_chunk = retrieved_data[0]
    #     for key, value in first_chunk.items():
    #         if key == 'chunk_content':
    #              print(f"  {key}: '{str(value)[:100]}...'")
    #         else:
    #              print(f"  {key}: {value}")

