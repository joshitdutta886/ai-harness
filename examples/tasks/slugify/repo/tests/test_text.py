import unittest

from textkit.text import slugify, title_case


class SlugifyTests(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(slugify("hello"), "hello")

    def test_collapses_and_strips(self):
        self.assertEqual(slugify("  Hello,   World!! "), "hello-world")

    def test_numbers(self):
        self.assertEqual(slugify("Top 10 Tips"), "top-10-tips")


class TitleCaseTests(unittest.TestCase):
    # Known failing, tracked separately (not part of the slug issue).
    def test_small_words(self):
        self.assertEqual(title_case("the lord of the rings"), "The Lord of the Rings")


if __name__ == "__main__":
    unittest.main()
