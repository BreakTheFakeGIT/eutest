import unittest
from text_eu_json import TextEuJson

class TestTextEuJson(unittest.TestCase):

    def test_remove_html(self):
        raw = "<p>Hello&nbsp;World</p>"
        result = TextEuJson(raw).remove_html().to_string()
        self.assertEqual(result.strip(), "Hello World")

    def test_remove_non_breaking_space(self):
        raw = "Hello World"
        result = TextEuJson(raw).remove_non_breaking_space().to_string()
        self.assertEqual(result, "Hello World")

    def test_trim_text(self):
        raw = "   Hello World   "
        result = TextEuJson(raw).trim_text().to_string()
        self.assertEqual(result, "Hello World")

    def test_full_processing_pipeline(self):
        raw = "<p>Hello&nbsp;World</p>\nThis is a test string."
        result = TextEuJson(raw).process().to_string()
        self.assertEqual(result, "Hello World This is a test string.")

if __name__ == "__main__":
    unittest.main()
