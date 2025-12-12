import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter #uv add langchain-text-splitters
#from langchain.text_splitter import RecursiveCharacterTextSplitter #uv add langchain-text-splitters
import math
from typing import Optional

from tests.text_interp import text
import src.utils.logger as logger_utils
from dotenv import load_dotenv
load_dotenv()


logger = logger_utils.setup_logger(name="eu_unified_text_cleaner")

class UnifiedTextCleaner:
    """
    Unified, immutable text cleaner with a fluent API that merges the features
    of TextEuJson and TextCleaner into one class.

    Each transformation returns a new instance. Choose between two pipelines via `process()`:
    - pipeline="eujson": remove HTML-like content, non-breaking spaces, and trim.
    - pipeline="full": lower-case, normalize spaces/newlines, remove sequences of dots, etc.
    """

    # Default HTML-like patterns from TextEuJson
    _DEFAULT_HTML_PATTERNS: List[str] = [
        r'<[^>]+>',       # HTML tags (assuming real HTML brackets)
        r'\\n',           # Literal \n sequences
        r'&nbsp;',        # Non-breaking space entity
        r'\\t',           # Literal \t sequences
        r'&lt;',          # Less-than entity
        r'&gt;',          # Greater-than entity
        # If your input really contains &amp;lt; etc, add those too:
        r'&amp;nbsp;',
        r'&amp;lt;',
        r'&amp;gt;'
    ]

    def __init__(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        self.text = text

    # -------------------------
    # Basic utilities / parity
    # -------------------------
    def to_string(self) -> str:
        """Return the current text as a string."""
        return self.text

    def lower_case(self) -> 'UnifiedTextCleaner':
        """Convert text to lowercase."""
        return UnifiedTextCleaner(self.text.lower())

    def trim(self) -> 'UnifiedTextCleaner':
        """Trim leading and trailing whitespace."""
        return UnifiedTextCleaner(self.text.strip())

    # -------------------------
    # HTML and escape handling
    # -------------------------
    def _sub_html(self, pattern: str) -> 'UnifiedTextCleaner':
        """
        Internal helper mirroring TextEuJson._sub_html:
        Replace matches with a single space and return a new instance.
        """
        new_text = re.sub(pattern, ' ', self.text)
        return UnifiedTextCleaner(new_text)

    def remove_html(self, patterns: Optional[List[str]] = None) -> 'UnifiedTextCleaner':
        """
        Remove common HTML tags and escape sequences by replacing them with spaces.
        You can override the default patterns by passing a custom list.
        """
        patterns = patterns or self._DEFAULT_HTML_PATTERNS
        new_text = self.text
        for pattern in patterns:
            new_text = re.sub(pattern, ' ', new_text)
        return UnifiedTextCleaner(new_text)

    def remove_non_breaking_space(self) -> 'UnifiedTextCleaner':
        """Remove Unicode non-breaking spaces (U+00A0)."""
        new_text = self.text.replace('\u00A0', '')
        return UnifiedTextCleaner(new_text)

    # -------------------------
    # Whitespace & newline ops
    # -------------------------
    def remove_all_whitespaces(self) -> 'UnifiedTextCleaner':
        """Replace multiple whitespace characters with a single space."""
        new_text = re.sub(r'\s+', ' ', self.text, flags=re.UNICODE)
        return UnifiedTextCleaner(new_text)

    def remove_newlines(self) -> 'UnifiedTextCleaner':
        """Remove newlines by joining lines with spaces."""
        new_text = ' '.join(self.text.splitlines())
        return UnifiedTextCleaner(new_text)

    def replace_newlines_with_spaces(self) -> 'UnifiedTextCleaner':
        """Replace newline characters with spaces."""
        return UnifiedTextCleaner(self.text.replace('\n', ' '))

    # -------------------------
    # Punctuation normalization
    # -------------------------
    def remove_seq_dot(self) -> 'UnifiedTextCleaner':
        """
        Remove sequences of two or more dots surrounded by spaces.
        Matches cases like: ' .. ', ' ... ', etc.
        """
        new_text = re.sub(r'\s(\.){2,}\s', ' ', self.text)
        return UnifiedTextCleaner(new_text)

    def remove_brackets(self, replace_with: str = '') -> 'UnifiedTextCleaner':
        """
        Convenience method to remove only bracket-like punctuation.
        (Parentheses, square/curly/angle quotes/brackets and similar.)

        If you already use remove_punctuation(), this is typically unnecessary.
        """
        # Common ASCII + Unicode bracket-like characters
        bracket_chars = '()[]{}<>«»‹›„“”‘’‚’❮❯⟨⟩「」『』【】〔〕〚〛〈〉《》'
        pattern = '[' + re.escape(bracket_chars) + ']'
        new_text = re.sub(pattern, replace_with, self.text)
        if replace_with.strip() == '':
            new_text = re.sub(r'\s+', ' ', new_text).strip()
        return UnifiedTextCleaner(new_text)

    def remove_punctuation(
            self,
            replace_with: str = '',
            keep: str = ''
        ) -> 'UnifiedTextCleaner':
            """
            Remove all Unicode punctuation characters, optionally keeping some,
            and optionally replacing removed punctuation with `replace_with`.
            """
            kept = set(keep)
            out = []
            for ch in self.text:
                if unicodedata.category(ch).startswith('P') and ch not in kept:
                    if replace_with:
                        out.append(replace_with)
                    # else: drop it
                else:
                    out.append(ch)

            new_text = ''.join(out)
            if replace_with.strip() == '':
                new_text = re.sub(r'\s+', ' ', new_text).strip()
            return UnifiedTextCleaner(new_text)

    def keep_punctuation(
        self,
        allowed: str,
        replace_removed_with: str = ''
    ) -> 'UnifiedTextCleaner':
        """
        Keep ONLY the punctuation characters listed in `allowed`,
        remove (or replace) all other punctuation. Non-punctuation characters
        (letters, digits, whitespace, symbols) are left untouched.
        """
        allowed_set = set(allowed)
        out = []
        for ch in self.text:
            if unicodedata.category(ch).startswith('P'):  # punctuation
                if ch in allowed_set:
                    out.append(ch)  # keep it
                else:
                    if replace_removed_with:
                        out.append(replace_removed_with)
                    # else: drop it
            else:
                out.append(ch)

        new_text = ''.join(out)
        # If we replaced with empty or whitespace, normalize spaces
        if replace_removed_with.strip() == '':
            new_text = re.sub(r'\s+', ' ', new_text).strip()
        return UnifiedTextCleaner(new_text)

    # -------------------------
    # Regex-based start index & cutting
    # -------------------------

    def index_text_start_word(
        self,
        pattern: str,
        text: Optional[str] = None,
        flags: int = re.IGNORECASE | re.UNICODE
    ) -> int:
        """
        Return the start index of the first match of `pattern` in `text`
        (or in self.text if `text` is None). If no match is found, return 0.
        """
        source = self.text if text is None else text
        try:
            match = re.search(pattern, source, flags=flags)
            return match.start() if match else 0
        except Exception:
            # Keep parity with original: return 0 on any failure
            return 0

    def cut_text_start_by_regex(
        self,
        stems: Optional[List[str]] = None,
        text: Optional[str] = None,
        flags: int = re.IGNORECASE | re.UNICODE
    ) -> 'UnifiedTextCleaner':
        """
        Build regex patterns from given stems (Polish phrases), find the earliest
        match in the text, and cut the text so it starts at that match.
        If no match is found, return the original text unchanged.
        """
        source = self.text if text is None else text

        # Default stems (deduplicated; retains the user's original intent)
        default_stems = [
            "opis stanu faktyczn",
            "opisu stanu faktyczn",
            "opisowi stanu faktyczn",
            "opisem stanu faktyczn",
            "opiśie stanu faktyczn",
            "opis zdarzenia przysz",
            "opisu zdarzenia przysz",
            "opisowi zdarzenia przysz",
            "opisem zdarzenia przysz",
            "opiśie zdarzenia przysz",
            "opis stanów",
            "opisu stanów",
            "opisowi stanów",
            "opisem stanów",
            "opiśie stanów",
            "opis stan",
            "opisem stan",
            "opisowi stan",
            "opiśie stan"
        ]

        stems = stems or default_stems

        # Build regexes: word boundary, stem with spaces as \s+, then \w* to match inflection
        regex_parts: List[str] = []
        for stem in stems:
            # Replace literal spaces with \s+ to match any whitespace sequences
            part = r'\b' + stem.replace(' ', r'\s+') + r'\w*'
            regex_parts.append(part)

        # Find start indices for each pattern
        start_indices: List[int] = []
        for reg in regex_parts:
            idx = self.index_text_start_word(pattern=reg, text=source, flags=flags)
            start_indices.append(idx)

        # Choose the earliest positive index (i.e., > 0). If all are 0, nothing matched.
        # Note: 0 may mean "found at start" or "not found" in original behavior,
        # so we check if any actual match exists first.
        actual_matches: List[int] = []
        for reg in regex_parts:
            m = re.search(reg, source, flags=flags)
            if m:
                actual_matches.append(m.start())

        if not actual_matches:
            # No matches -> return original
            return UnifiedTextCleaner(source)

        cut_at = min(actual_matches)

        # Cut from earliest match position
        new_text = source[cut_at:]

        return UnifiedTextCleaner(new_text)

    def _build_regex_from_stems(self, stems: List[str]) -> List[str]:
        """
        Internal helper: convert phrase stems into regex parts:
        - word boundary
        - replace spaces with \\s+ (any whitespace)
        - allow word continuation with \\w* (to cover inflection)
        """
        regex_parts: List[str] = []
        for stem in stems:
            part = r'\b' + stem.replace(' ', r'\s+') + r'\w*'
            regex_parts.append(part)
        return regex_parts

    def index_char_in_text(
        self,
        char: str,
        text: Optional[str] = None,
        flags: int = re.IGNORECASE | re.UNICODE,
        anchor: str = "first",
        include_match: bool = False
    ) -> int:
        """
        Return the end index of the first match of `char` in `text`
        (or in self.text if `text` is None). If no match is found, return .
        """
        source = self.text if text is None else text
        try:
            matches = list(re.finditer(re.escape(char), source, flags=flags))
            # Pick the anchor match (first occurrence or last occurrence)
            if anchor not in ("first", "last"):
                raise ValueError('anchor must be "first" or "last"')

            chosen = min(matches, key=lambda m: m.start()) if anchor == "first" else max(matches, key=lambda m: m.start())

            if include_match:
                # Keep everything up to and including the match
                end_idx = chosen.end()
            else:
                # Keep everything strictly before the match
                end_idx = chosen.start()

            # Guard against negative or out-of-range (shouldn't happen, but be safe)
            return max(0, min(end_idx, len(source)))
        except Exception:
            # Keep parity with original: return len(source) on any failure
            return len(source)

    def cut_text_end_by_regex(
        self,
        stems: Optional[List[str]] = None,
        text: Optional[str] = None,
        flags: int = re.IGNORECASE | re.UNICODE,
        anchor: str = "first",
        include_match: bool = False
    ) -> 'UnifiedTextCleaner':
        """
        Cut the text at an END marker determined by regex stems.
        """
        source = self.text if text is None else text

        # Reasonable defaults for common "end" sections in PL documents (tweak as needed)
        default_stems = [
            'poucze',
            'ocena stanowi',
            'ocenie stanowi',
            'informacja zakres',
            'podstawa prawna',
            'stronie przysługuje prawo',
            'dodatkowe inform',
            'informacja o zakres',
            'uzasadnienie interpr',
            'postępowanie przed sądami administ',
            'zażalenie na postan',
            "stanowisko w sprawie",
            "stanowisk w sprawie",
            "stanowiskiem w sprawie"
            "pani stanowisko",
            "pana stanowisko",
            "państwo stanowisko",
        ]
        stems = stems or default_stems

        regex_parts = self._build_regex_from_stems(stems)

        # Collect all matches across all patterns
        matches = []
        for reg in regex_parts:
            for m in re.finditer(reg, source, flags=flags):
                matches.append(m)

        if not matches:
            # No end markers -> return original unchanged
            return UnifiedTextCleaner(source)

        # Pick the anchor match (first occurrence or last occurrence)
        if anchor not in ("first", "last"):
            raise ValueError('anchor must be "first" or "last"')

        chosen = min(matches, key=lambda m: m.start()) if anchor == "first" else max(matches, key=lambda m: m.start())

        if include_match:
            # Keep everything up to and including the match
            end_idx = chosen.end()
        else:
            # Keep everything strictly before the match
            end_idx = chosen.start()

        # Guard against negative or out-of-range (shouldn't happen, but be safe)
        end_idx = max(0, min(end_idx, len(source)))

        new_text = source[:end_idx]
        return UnifiedTextCleaner(new_text)

    # -------------------------
    # Pipelines
    # -------------------------
    def process(self, pipeline: str = "eu_start_end") -> 'UnifiedTextCleaner':
        if pipeline == "eu_start_end":
            return (
                self.cut_text_start_by_regex()
                .cut_text_end_by_regex()
                .to_string()
                )
        elif pipeline == "normalized":
            return (
                self.lower_case()
                    .remove_html()
                    .remove_non_breaking_space()
                    .trim()
                    .replace_newlines_with_spaces()
                    .remove_all_whitespaces()
                    .remove_seq_dot()
                    .remove_newlines()
                    .keep_punctuation(allowed="().?%", replace_removed_with=' ') \
                    .to_string()
            )
        else:
            raise ValueError('Unknown pipeline. Use "full" or "eujson".')


def stat_text(text: str) -> Dict[str,Any]:
        len_text = len(text)
        cnt_newlines_text = text.count('\n')
        cnt_words_text = len(text.split())
        cnt_pages_text = int(round(len_text/4420))+1
        cnt_words_text_per_page = int(round(cnt_words_text/cnt_pages_text))
        cnt_question_marks = text.count('?')
        return {'len_text': len_text,
                'cnt_pages_text': cnt_pages_text,
                'cnt_newlines_text': cnt_newlines_text,
                'cnt_words_text': cnt_words_text,
                'cnt_words_text_per_page': cnt_words_text_per_page,
                'cnt_question_marks': cnt_question_marks,
                }

def text_split_fact_q(text: str) -> Tuple[str, str]:
    text_q = (UnifiedTextCleaner(text=text).cut_text_start_by_regex(stems=["pytani"]).to_string())
    idx_q_end = UnifiedTextCleaner(text=text_q).index_char_in_text(char='?', anchor='last', include_match=True)
    text_q = text_q[:idx_q_end]
    text_fact = text.replace(text_q, "", 1)
    text_fact = text_fact[:8840]
    text_q = text_q[:4420]
    if not text_q:
        text_q = None
        raise ValueError("No questions found in the text.")
    return text_fact, text_q

def full_process(text: str) -> Tuple[str, Optional[str]]:
    text_start_end = UnifiedTextCleaner(text=text).process(pipeline='eu_start_end')
    text_norm = UnifiedTextCleaner(text=text_start_end).process(pipeline='normalized')
    return text_start_end, text_norm

def extract_fact_q(text: str) -> Tuple[str, Optional[str]]:
    try:
        text_norm_fact, text_norm_q= text_split_fact_q(text=text) # 2 and 1 word's pagess
    except ValueError as e:
        logger.error(e)
    return text_norm_fact, text_norm_q








class ChunkPlanner:
    """
    Plan and analyze fixed-size chunking with overlap over a linear sequence
    """

    def __init__(self, length: int, chunk_size: Optional[int] = None, overlap: int = 0) -> None:
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer.")
            if overlap >= chunk_size:
                raise ValueError("overlap must be strictly less than chunk_size.")

        self.length = length
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ---------- Core formulas ----------

    @staticmethod
    def num_chunks_static(length: int, chunk_size: int, overlap: int) -> int:
        """Static version: N = ceil((L - O) / (C - O))."""
        if length <= 0 or chunk_size <= 0:
            raise ValueError("length and chunk_size must be positive.")
        if not (0 <= overlap < chunk_size):
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size.")
        step = chunk_size - overlap
        if chunk_size >= length:
            return 1
        return math.ceil((length - overlap) / step)

    def num_chunks(self) -> int:
        """Compute number of chunks using current planner settings."""
        if self.chunk_size is None:
            raise ValueError("chunk_size is not set.")
        return self.num_chunks_static(self.length, self.chunk_size, self.overlap)

    @staticmethod
    def max_chunk_size(length: int, overlap: int, max_chunks: int) -> int:
        """
        Maximize chunk size C subject to N <= max_chunks.
        Derived: C = (L - O) / N_max + O, then floor to integer.
        """
        if length <= 0:
            raise ValueError("length must be a positive integer.")
        if overlap < 0:
            raise ValueError("overlap must be a non-negative integer.")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be a positive integer.")

        c_cont = (length - overlap) / max_chunks + overlap
        c_max = math.floor(c_cont)
        if c_max <= overlap:
            raise ValueError(
                "No valid chunk_size exists (computed C_max <= overlap). "
                "Increase max_chunks or decrease overlap."
            )
        return c_max

    @staticmethod
    def max_overlap_for_Nmax(length: int, chunk_size: int, Nmax: int) -> int:
        """
        Largest integer overlap O in [0, C-1] such that N <= Nmax.
        Uses bound: O <= (Nmax*C - L) / (Nmax - 1) for Nmax >= 2.
        Special case Nmax=1: requires C >= L; then O_max = C-1.
        """
        if length <= 0 or chunk_size <= 0 or Nmax <= 0:
            raise ValueError("length, chunk_size, Nmax must be positive.")
        if Nmax == 1:
            if chunk_size < length:
                raise ValueError("Nmax=1 impossible unless chunk_size >= length.")
            return chunk_size - 1

        bound = (Nmax * chunk_size - length) / (Nmax - 1)
        Omax = math.floor(bound)
        Omax = max(0, min(Omax, chunk_size - 1))
        # Verify
        while Omax >= 0 and ChunkPlanner.num_chunks_static(length, chunk_size, Omax) > Nmax:
            Omax -= 1
        if Omax < 0:
            raise ValueError("No non-negative overlap satisfies N <= Nmax.")
        return Omax

    @staticmethod
    def overlaps_for_exact_N(length: int, chunk_size: int, N: int) -> List[int]:
        """
        Return all integer overlaps O in [0, C-1] that yield exactly N chunks.
        Uses inequality interval; falls back to checking when N < 3.
        """
        if length <= 0 or chunk_size <= 0 or N <= 0:
            raise ValueError("length, chunk_size, N must be positive.")

        valid: List[int] = []
        if N >= 3:
            lower = ((N - 1) * chunk_size - length) / (N - 2)  # O > lower
            upper = (N * chunk_size - length) / (N - 1)        # O <= upper
            start = max(0, math.floor(lower) + 1)
            end = min(chunk_size - 1, math.floor(upper))
            for O in range(start, end + 1):
                if ChunkPlanner.num_chunks_static(length, chunk_size, O) == N:
                    valid.append(O)
            return valid

        # Edge cases N=1 or N=2
        for O in range(0, chunk_size):
            if ChunkPlanner.num_chunks_static(length, chunk_size, O) == N:
                valid.append(O)
        return valid

    # ---------- Indices & splitting ----------

    @staticmethod
    def chunk_indices(length: int, chunk_size: int, overlap: int) -> List[Tuple[int, int]]:
        """
        Return start/end indices (end-exclusive) for each chunk covering [0, length).
        """
        n = ChunkPlanner.num_chunks_static(length, chunk_size, overlap)
        step = chunk_size - overlap
        indices: List[Tuple[int, int]] = []
        start = 0
        for _ in range(n):
            end = min(start + chunk_size, length)
            indices.append((start, end))
            start += step
            if start >= length:
                break
        return indices

    def indices(self) -> List[Tuple[int, int]]:
        """Instance method variant using current settings."""
        if self.chunk_size is None:
            raise ValueError("chunk_size is not set.")
        return self.chunk_indices(self.length, self.chunk_size, self.overlap)

    @staticmethod
    def split_sequence(seq: Iterable, chunk_size: int, overlap: int) -> List[List]:
        """
        Split a generic sequence into overlapping chunks using indices.
        Works for lists, strings, etc.
        """
        if hasattr(seq, "__len__"):
            length = len(seq)  # type: ignore
        else:
            raise TypeError("seq must be a sized iterable (e.g., list or string).")

        ranges = ChunkPlanner.chunk_indices(length, chunk_size, overlap)
        result: List[List] = []
        # Convert to list for slicing generically
        seq_list = list(seq)
        for s, e in ranges:
            result.append(seq_list[s:e])
        return result

    def split_text(self, text: str) -> List[str]:
        """
        Split a string 'text' using current planner's C and O.
        """
        if self.chunk_size is None:
            raise ValueError("chunk_size is not set.")
        # Use indices to slice text
        parts = []
        for s, e in self.indices():
            parts.append(text[s:e])
        return parts

    # ---------- Convenience setters ----------

    def set_chunk_size(self, chunk_size: int) -> None:
        """Update chunk_size with validation against current overlap."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if self.overlap >= chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")
        self.chunk_size = chunk_size

    def set_overlap(self, overlap: int) -> None:
        """Update overlap with validation against current chunk_size (if set)."""
        if overlap < 0:
            raise ValueError("overlap must be non-negative.")
        if self.chunk_size is not None and overlap >= self.chunk_size:
            raise ValueError("overlap must be strictly less than chunk_size.")
        self.overlap = overlap



if __name__ == "__main__":
    # Example usage

    def print_text(text:str) -> None:
        print(text)
        logger.info(f'{stat_text(text=text)}')
    print(text)

    text_start_end, text_norm = full_process(text=text)
    text_norm_fact, text_norm_q = extract_fact_q(text=text_norm)
    if text_norm_q:
        _, text_q = extract_fact_q(text=text_start_end)
        list_of_questions = [q.lower() for q  in text_q.split('\n') if len(q) > 21]
        if list_of_questions:
            #zip(['pytania']*len(list_of_questions), list_of_questions)
            print("="*10+' START TEXT '+"="*50)
            print(list_of_questions)
            print("="*10+' END TEXT '+"="*50)
            # TODO embeddings
            # TODO zip
            # TODO save to postgres

    # Example: plan for a text of length 1000, with chunk_size=200 and overlap=50
    len_text = len(text_norm)
    chunk_size = [x*7 for x in [300,550,750]]
    overlap = 50
    for 
    planner = ChunkPlanner(length=len_text, chunk_size=1750, overlap=50)

    print("N =", planner.num_chunks())
    print("Indices:", planner.indices())

    # Maximize chunk size subject to N_max
    # Cmax = ChunkPlanner.max_chunk_size(length=len_text, overlap=50, max_chunks=7)
    # print("C_max for N_max=7:", Cmax)

    # # Max overlap allowed for N_max
    # Omax = ChunkPlanner.max_overlap_for_Nmax(length=len_text, chunk_size=200, Nmax=7)
    # print("O_max for N_max=7:", Omax)

    # # Which overlaps yield exactly N=7?
    # overlaps = ChunkPlanner.overlaps_for_exact_N(length=len_text, chunk_size=200, N=7)
    # print("Overlaps for exactly N=7:", overlaps)

    # Split actual text
    planner2 = ChunkPlanner(length=len(len_text), chunk_size=1750, overlap=750)
    parts = planner2.split_text(text)
    print(f"Produced {len(parts)} chunks; first chunk length:", len(parts[0]))



    def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 1750) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                  chunk_overlap=chunk_overlap,
                                                  length_function=len,)
        return splitter.split_text(text)

    # for text in [text_norm_fact, text_norm_q, text_q]:
    #     print("="*10 + " START TEXT " + "="*50)
    #     print_text(text=text)
    #     print("="*10 + " END TEXT " + "="*50)



