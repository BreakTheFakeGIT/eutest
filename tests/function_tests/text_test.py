# Original dictionary
my_dict = {'old_key': 'value'}

# Rename 'old_key' to 'new_key'
my_dict['new_key'] = my_dict.pop('old_key')

print(my_dict)

def rename_keys(data: dict, key_map: dict) -> dict:
    """
    Rename keys in a dictionary using a mapping.

    Parameters:
    - data: The original dictionary.
    - key_map: A dictionary where keys are old keys and values are new keys.

    Returns:
    - A new dictionary with keys renamed.
    """
    return {key_map.get(k, k): v for k, v in data.items()}
original = {'name': 'Alice', 'age': 30, 'city': 'Warsaw'}
mapping = {'name': 'full_name', 'city': 'location'}

renamed = rename_keys(original, mapping)
print(renamed)

def rename_keys_nested(data: dict, key_map: dict) -> dict:
    """
    Recursively rename keys in a nested dictionary using a mapping.

    Parameters:
    - data: The original dictionary (can be nested).
    - key_map: A dictionary where keys are old keys and values are new keys.

    Returns:
    - A new dictionary with keys renamed.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = key_map.get(k, k)
            new_dict[new_key] = rename_keys_nested(v, key_map)
        return new_dict
    elif isinstance(data, list):
        return [rename_keys_nested(item, key_map) for item in data]
    else:
        return data

original = {
    'name': 'Alice',
    'details': {
        'age': 30,
        'city': 'Warsaw',
        'contact': {
            'email': 'alice@example.com',
            'phone': '123456789'
        }
    }
}

mapping = {
    'name': 'full_name',
    'city': 'location',
    'email': 'email_address'
}

renamed = rename_keys_nested(original, mapping)
print(renamed)

# {
#     'full_name': 'Alice',
#     'details': {
#         'age': 30,
#         'location': 'Warsaw',
#         'contact': {
#             'email_address': 'alice@example.com',
#             'phone': '123456789'
#         }
#     }
# }


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



@dataclass(frozen=True)
class Text2:
    """
    Immutable text processing object with reversible, chainable pipeline.

    - Each transformation returns a NEW instance (immutability).
    - Keeps `history` of step names and `snapshots` of text BEFORE each step.
    - Supports undo operations returning NEW instances.
    - JSON (de)serialization for portability.
    - Optional immutable metadata.

    Fields:
        text: Current text state.
        history: Tuple of step names applied in order.
        snapshots: Tuple of text states BEFORE each corresponding step.
        meta: Immutable metadata mapping (wrapped by MappingProxyType).

    Example:
        obj = TextEuJson.from_text("  Hello...\n\nWorld!!   ")
        obj2 = obj.pipeline()
        print(obj2.text)  # "hello... world!!"
        obj3 = obj2.undo_last()
        print(obj3.text)  # Text state before the last step
        raw = obj2.to_json()
        restored = TextEuJson.from_json(raw)
    """

    text: str
    history: Tuple[str, ...] = field(default_factory=tuple)
    snapshots: Tuple[str, ...] = field(default_factory=tuple)
    meta: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    # --- Construction helpers -------------------------------------------------

    @classmethod
    def from_text(cls, text: Any, meta: Optional[Dict[str, Any]] = None) -> TextEuJson:
        """Create instance from any object (cast to str), with optional metadata."""
        m = MappingProxyType(dict(meta)) if meta else MappingProxyType({})
        return cls(text=str(text), history=tuple(), snapshots=tuple(), meta=m)

    def with_meta(self, **updates: Any) -> TextEuJson:
        """Return a new instance with merged immutable metadata."""
        new_meta = dict(self.meta)
        new_meta.update(updates)
        return replace(self, meta=MappingProxyType(new_meta))

    # --- Internal helper to apply a step immutably ----------------------------

    def _next(self, step_name: str, func, logger: Optional[logging.Logger] = None) -> TextEuJson:
        """Apply `func(self.text)` and return a NEW instance with updated history/snapshot."""
        before = self.text
        after = func(before)
        if logger:
            logger.info(f"Applied step: {step_name} | before='{before}' | after='{after}'")
        return replace(
            self,
            text=after,
            history=self.history + (step_name,),
            snapshots=self.snapshots + (before,),
        )

    # --- Cleaning steps (each returns NEW instance) ----------------------------

    def to_string(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        return self._next("to_string", lambda t: str(t), logger)

    def lower_case_string(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        return self._next("lower_case_string", lambda t: t.lower(), logger)

    def text_trim(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        return self._next("text_trim", lambda t: t.strip(), logger)

    def replace_newlines_with_spaces(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        return self._next("replace_newlines_with_spaces", lambda t: t.replace("\n", " "), logger)

    def remove_all_whitespaces(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        return self._next("remove_all_whitespaces", lambda t: re.sub(r"\s+", " ", t, flags=re.UNICODE), logger)

    def remove_seq_dot(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        # Replace sequences of 2+ dots that are surrounded by spaces with a single space.
        pattern = r"\s(\.){2,}\s"
        return self._next("remove_seq_dot", lambda t: re.sub(pattern, " ", t), logger)

    def remove_newline(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        return self._next("remove_newline", lambda t: " ".join(t.splitlines()), logger)

    # --- Combined pipeline (returns NEW instance) ------------------------------

    def pipeline(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        """Apply the full cleaning pipeline in order, returning a NEW instance."""
        if logger:
            logger.info("Starting pipeline...")
        out = (
            self.to_string(logger)
                .lower_case_string(logger)
                .text_trim(logger)
                .replace_newlines_with_spaces(logger)
                .remove_all_whitespaces(logger)
                .remove_seq_dot(logger)
                .remove_newline(logger)
        )
        if logger:
            logger.info("Pipeline completed.")
        return out

    # --- Reversibility (returns NEW instances) --------------------------------

    def undo_last(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        """Return a NEW instance representing the state before the last step."""
        if not self.history:
            raise ValueError("No steps to undo.")
        if logger:
            logger.info(f"Undoing step: {self.history[-1]}")
        # Restore last snapshot as text and drop last elements from history/snapshots
        restored_text = self.snapshots[-1]
        return replace(self, text=restored_text, history=self.history[:-1], snapshots=self.snapshots[:-1])

    def undo_all(self, logger: Optional[logging.Logger] = None) -> TextEuJson:
        """Return a NEW instance representing the original state (before any steps)."""
        if not self.history:
            # Already original
            return self
        original_text = self.snapshots[0]
        if logger:
            logger.info("Undoing all steps to original state.")
        return replace(self, text=original_text, history=tuple(), snapshots=tuple())

    # --- Introspection ---------------------------------------------------------

    @property
    def original_text(self) -> str:
        """Text before any transformations (if steps exist; else current)."""
        return self.snapshots[0] if self.snapshots else self.text

    def get_history(self) -> Tuple[str, ...]:
        """Tuple of applied step names in order."""
        return self.history

    # --- Serialization ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "text": self.text,
            "history": list(self.history),
            "snapshots": list(self.snapshots),
            "meta": dict(self.meta),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TextEuJson:
        """Deserialize from dict; enforces immutability on meta."""
        text = str(data.get("text", ""))
        history = tuple(data.get("history", []))
        snapshots = tuple(data.get("snapshots", []))
        meta = MappingProxyType(dict(data.get("meta", {})))
        return cls(text=text, history=history, snapshots=snapshots, meta=meta)

    @classmethod
    def from_json(cls, raw: str) -> TextEuJson:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(raw))

    # --- Convenience -----------------------------------------------------------

    def __call__(self) -> str:
        """Return current text (callable convenience)."""
        return self.text








import re
import os
from typing import Callable, List, Tuple
from dataclasses import dataclass
import src.utils.logger as logger_utils
class ReversibleTextCleaner:
    """Text cleaner with reversible pipeline functionality."""

    def __init__(self):
        self._text = None
        self._history: List[Tuple[str, str]] = []  # (step_name, previous_text)

    def set_text(self, text: str):
        """Initialize text and clear history."""
        self._text = str(text)
        self._history.clear()

    def _apply_step(self, func: Callable[[str], str], step_name: str):
        """Apply a cleaning step and record history."""
        if self._text is None:
            raise ValueError("No text set. Use set_text() first.")
        previous = self._text
        self._text = func(self._text)
        self._history.append((step_name, previous))

    # Cleaning steps
    def lower_case_string(self):
        self._apply_step(lambda t: t.lower(), "lower_case_string")

    def text_trim(self):
        self._apply_step(lambda t: t.strip(), "text_trim")

    def replace_newlines_with_spaces(self):
        self._apply_step(lambda t: t.replace("\n", " "), "replace_newlines_with_spaces")

    def remove_all_whitespaces(self):
        self._apply_step(lambda t: re.sub(r"\s+", " ", t, flags=re.UNICODE), "remove_all_whitespaces")

    def remove_seq_dot(self):
        self._apply_step(lambda t: re.sub(r"\s(\.){2,}\s", " ", t), "remove_seq_dot")

    def remove_newline(self):
        self._apply_step(lambda t: " ".join(t.splitlines()), "remove_newline")

    # Pipeline execution
    def apply_pipeline(self):
        """Run all cleaning steps in order."""
        self.lower_case_string()
        self.text_trim()
        self.replace_newlines_with_spaces()
        self.remove_all_whitespaces()
        self.remove_seq_dot()
        self.remove_newline()

    # Reversal methods
    def undo_last(self):
        """Undo the last cleaning step."""
        if not self._history:
            raise ValueError("No steps to undo.")
        step_name, previous_text = self._history.pop()
        self._text = previous_text
        return step_name

    def undo_all(self):
        """Undo all steps and restore original text."""
        if not self._history:
            raise ValueError("No steps to undo.")
        _, original_text = self._history[0]
        self._text = original_text
        self._history.clear()

    def get_text(self) -> str:
        """Get current text state."""
        return self._text

    def get_history(self) -> List[str]:
        """Get list of applied steps."""
        return [step for step, _ in self._history]



import re


@dataclass(frozen=True)
class TextCleaner:
    """
    Immutable text cleaner with fluent API.
    Each method returns a new instance with transformed text.
    """
    text: str

    def to_string(self) -> 'TextCleaner':
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

    def simple_clean_text(self) -> 'TextCleaner':
        """
        Apply full cleaning pipeline in one call.
        """
        return (self.to_string()
                    .lower_case()
                    .trim()
                    .replace_newlines_with_spaces()
                    .remove_all_whitespaces()
                    .remove_seq_dot()
                    .remove_newlines())

    def get(self) -> str:
        """Return the cleaned text."""
        return self.text
    

    async def fetch_texts_batch_fetchmany(sql: str, limit: int = 100, tax_type: str | None = None, batch_size_sql: int = 10) -> List[Dict[str, Any]]:
    params: Tuple[Any, ...] = ()
    if tax_type:
        sql += " AND typ_podatku = %s"
        params = (tax_type,)
    sql += " ORDER BY id_informacji DESC LIMIT %s"
    params = params + (limit, )

    async with await psycopg.AsyncConnection.connect(PG_CONN_STR) as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            while True:
                rows = await cur.fetchmany(batch_size_sql)
                logger.info(f"Processing fetch_texts_batch_fetchmany {len(rows)} completed.")
                if not rows:
                    break
                return rows
            

def split_text_by_header(text: str) -> str:
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
