import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze import clean_text, is_letter_or_number, tokenize
import jieba


class PreprocessingTests(unittest.TestCase):
    def test_separators_do_not_join_words(self):
        self.assertEqual(clean_text("甲，乙。丙\n丁\t戊★己\u200b庚"), "甲 乙 丙 丁 戊 己 庚")
        tokenizer = jieba.Tokenizer()
        tokenizer.tmp_dir = str(Path(__file__).resolve().parents[1] / ".cache")
        self.assertEqual(tokenize("张，三", tokenizer), ["张", "三"])

    def test_unicode_letters_and_numbers(self):
        for char in "汉字ＡαⅧ３二":
            self.assertTrue(is_letter_or_number(char), repr(char))
        for char in "，。★😀\n \u0301":
            self.assertFalse(is_letter_or_number(char), repr(char))


if __name__ == "__main__":
    unittest.main()
