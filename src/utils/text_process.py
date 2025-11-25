import re
import os
import src.utils.logger as logger_utils

#logger = logger_utils.setup_logger(name='src_utils_text_process')

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
