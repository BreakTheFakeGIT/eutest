import re
from typing import List, Dict, Type 
from itertools import product
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from src.utils.text_process import TextEuJson
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


###################################
logger = logger_utils.setup_logger(name=f'eu_text_splitter')

###################################
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
        text = cut_text_word(text=text, pattern=reg)
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

def process_cut_text(text: str) -> str:
    # try:
    #     text = TextEuJson(text).process().to_string()
    # except Exception as e:
    #     raise logger.error("Error in TextEuJson processing: {}".format(e))
    try:
        text = cut_text_start_by_regex(text)
        text = cut_text_by_regex(text)
        return text
    except Exception as e:
        raise logger.error("Invalid mode. cut_text_start_by_regex or cut_text_by_regex".format(e))



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

def text_splitting(
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
        all_chunks = [TextEuJson(chunk).process().to_string() for chunk in all_chunks if len(chunk.strip()) > 50]
        results[experiment_key] = all_chunks
        logger.info(f"Chunks after: {len(all_chunks)}")

    return results