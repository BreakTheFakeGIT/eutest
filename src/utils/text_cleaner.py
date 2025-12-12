import re
import unicodedata
from typing import List, Tuple, Optional
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


def pipeline_process(text: str) -> Tuple[str, Optional[str]]:
    text_start_end = UnifiedTextCleaner(text=text).process(pipeline='eu_start_end')
    text_norm = UnifiedTextCleaner(text=text_start_end).process(pipeline='normalized')
    return text_start_end, text_norm

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

def extract_fact_q(text: str) -> Tuple[str, Optional[str]]:
    try:
        text_norm_fact, text_norm_q= text_split_fact_q(text=text) # 2 and 1 word's pagess
    except ValueError as e:
        logger.error(e)
    return text_norm_fact, text_norm_q