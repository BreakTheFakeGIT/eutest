import re
import os
from src.logger import setup_logger

logger = setup_logger(name='text')


class TextProcessor:
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

    def data_processing(self):
        """ Function """
        logger.info(f"Method: {self.data_processing.__name__} - start data processing completed.")
        self.to_string()
        self.remove_html()
        self.remove_bs4_xa0()
        self.trim_text()
        logger.info(f"Method: {self.data_processing.__name__} - end data processing completed.")
        return self.text

if __name__ == "__main__":
    sample_text = "<p>This is a sample text with <b>HTML</b> tags &nbsp; and some &lt;entities&gt;.</p>\n"
    processor = TextProcessor(sample_text)
    cleaned_text = processor.data_processing()
    print("Original Text:", sample_text)
    print("Cleaned Text:", cleaned_text)

