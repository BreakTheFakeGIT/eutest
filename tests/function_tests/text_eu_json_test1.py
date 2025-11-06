import re
import os
import logging


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
        processed = (
            self.remove_html()
                .remove_non_breaking_space()
                .trim_text()
        )
        return processed


class TextEuJson2:
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        self.text = text

    def to_string(self) -> str:
        """Returns the text as a string."""
        return self.text

    def _sub_html(self, pattern: str) -> str:
        """Replaces HTML-like patterns in the text with a space."""
        return re.sub(pattern, ' ', self.text)

    def remove_html(self) -> str:
        """Removes common HTML tags and escape sequences."""
        patterns = [
            r'<[^>]+>',       # HTML tags
            r'\\n',           # Newlines
            r'&nbsp;',        # Non-breaking space
            r'\\t',           # Tabs
            r'&lt;',          # Less-than
            r'&gt;'           # Greater-than
        ]
        for pattern in patterns:
            self.text = self._sub_html(pattern)
        return self.text

    def remove_non_breaking_space(self) -> str:
        """Removes non-breaking space characters."""
        self.text = self.text.replace(u'\xa0', u'')
        return self.text

    def trim_text(self) -> str:
        """Trims leading and trailing whitespace."""
        self.text = self.text.strip()
        return self.text

    def process(self) -> str:
        """Runs all text cleaning steps."""
        self.remove_html()
        self.remove_non_breaking_space()
        self.trim_text()
        return self.text


class TextEuJson3:
    def __init__(self, text: str):
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
        self.text = text

    def to_string(self) -> str:
        """ Function """
        return str(self.text)

    def sub_html(self, regex: str) -> str:
        """ Function """
        return regex.sub(' ', self.text)

    def remove_html(self) -> str:
        """ Function """
        for char in (r'<[^>]+>', r'\\n', r'&nbsp;',r'\\t',r'&lt;',r'&gt;'):
            regex = re.compile(char)
            self.text = self.sub_html(regex)
        return self.text

    def remove_bs4_xa0(self) -> str:
        """ Function """
        return self.text.replace(u'\xa0', u'')

    def trim_text(self) -> str:
        """ Function """
        return self.text.strip()

    def process(self):
        """ Function """
        self.to_string()
        self.remove_html()
        self.remove_bs4_xa0()
        self.trim_text()
        return self.text


if __name__ == "__main__":

    raw_text = "<p>Hello&nbsp;World\n\n</p>\n"
    text_obj = TextEuJson(raw_text)
    cleaned_text = text_obj.process().to_string()
    print(cleaned_text)

    text_obj2 = TextEuJson2(raw_text)
    cleaned_text2 = text_obj2.process()
    print(cleaned_text2)

    text_obj3 = TextEuJson3(raw_text)
    cleaned_text3 = text_obj3.process()
    print(cleaned_text3)
