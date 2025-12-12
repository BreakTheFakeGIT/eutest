import re
from typing import List, Dict, Tuple
from itertools import product
from langchain.text_splitter import RecursiveCharacterTextSplitter
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()

###################################
logger = logger_utils.setup_logger(name=f'eu_text_splitter')

###################################
class TextEuJson:
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        self.text = text

    def to_string(self) -> str:
        """Returns the text as a string."""
        return self.text

    def _sub_html(self, pattern: str) -> 'TextEuJson':
        """Returns a new instance with HTML-like patterns replaced by a space."""
        new_text = re.sub(pattern, ' ', self.text)
        return TextEuJson(new_text)

    def remove_html(self) -> 'TextEuJson':
        """Returns a new instance with common HTML tags and escape sequences removed."""
        patterns = [
            r'<[^>]+>',       # HTML tags
            r'\\n',           # Newlines
            r'&nbsp;',        # Non-breaking space
            r'\\t',           # Tabs
            r'&lt;',          # Less-than
            r'&gt;'           # Greater-than
        ]
        new_text = self.text
        for pattern in patterns:
            new_text = re.sub(pattern, ' ', new_text)
        return TextEuJson(new_text)

    def remove_non_breaking_space(self) -> 'TextEuJson':
        """Returns a new instance with non-breaking space characters removed."""
        new_text = self.text.replace(u'\xa0', u'')
        return TextEuJson(new_text)

    def trim_text(self) -> 'TextEuJson':
        """Returns a new instance with leading and trailing whitespace removed."""
        new_text = self.text.strip()
        return TextEuJson(new_text)

    def process(self) -> 'TextEuJson':
        """Runs all text cleaning steps and returns a new processed instance."""
        #logger.info(f"Method: {self.process.__name__} - start data processing completed.")
        processed = (
            self.remove_html()
                .remove_non_breaking_space()
                .trim_text()
        )
        #logger.info(f"Method: {self.process.__name__} - end data processing completed.")
        return processed

class TextCleaner:
    """
    Immutable text cleaner with fluent API.
    Each method returns a new instance with transformed text.
    """
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        self.text = text

    def get(self) -> 'TextCleaner':
        """Convert to string."""
        return TextCleaner(str(self.text))

    def lower_case(self) -> 'TextCleaner':
        """Convert text to lowercase."""
        return TextCleaner(self.text.lower())

    def trim(self) -> 'TextCleaner':
        """Trim leading and trailing whitespace."""
        return TextCleaner(self.text.strip())

    def remove_seq_dot(self) -> 'TextCleaner':
        """Remove sequences of two or more dots surrounded by spaces."""
        new_text = re.sub(r'\s(\.){2,}\s', ' ', self.text)
        return TextCleaner(new_text)

    def remove_all_whitespaces(self) -> 'TextCleaner':
        """Replace multiple whitespace characters with a single space."""
        new_text = re.sub(r'\s+', ' ', self.text, flags=re.UNICODE)
        return TextCleaner(new_text)

    def remove_newlines(self) -> 'TextCleaner':
        """Remove newlines by joining lines with spaces."""
        new_text = ' '.join(self.text.splitlines())
        return TextCleaner(new_text)

    def replace_newlines_with_spaces(self) -> 'TextCleaner':
        """Replace newline characters with spaces."""
        return TextCleaner(self.text.replace('\n', ' '))

    def process(self) -> 'TextCleaner':
        """
        Apply full cleaning pipeline in one call.
        """
        return (self.get()
                    .lower_case()
                    .trim()
                    .replace_newlines_with_spaces()
                    .remove_all_whitespaces()
                    .remove_seq_dot()
                    .remove_newlines())

    def to_string(self) -> str:
        """Return the cleaned text."""
        return self.text

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

def text_splitting(
    text: str,
    chunk_sizes: List[int],
    chunk_overlaps: List[int],
) -> Dict[str, List[str]]:
    """
    Splits text into chunks using different parameters and allows filtering.
    """
    results = {}
    for size_overlap in list(product(chunk_sizes,chunk_overlaps, repeat=1)):

        size, overlap = size_overlap
        if overlap >= size:
            logger.info(f"Skipping experiment: overlap ({overlap}) >= size ({size})")
            continue

        experiment_key = f"size_{size}_overlap_{overlap}"
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        except Exception as e:
            logger.info(f"Error initializing splitter for {experiment_key}: {e}")
            continue

        results[experiment_key] = text_splitter.split_text(text)
    return results

def process_entire_text(text: str) -> str:
    text = TextEuJson(text=text).process().to_string()
    text = TextCleaner(text=text).process().to_string()
    return text

def process_chunk_text(text: str,chunk_size: List[int], chunk_overlap: List[int] ) -> List[str]:
    text = process_entire_text(text=text)
    text = process_cut_text(text=text)
    chunks = text_splitting(text=text, chunk_sizes= chunk_size, chunk_overlaps=chunk_overlap)
    return chunks