from unittest import TestCase

from helpers.textual_analysis import *


class Test(TestCase):
    def test_count_common_words(self):
        expected = 0.75
        result = count_common_words("I am here yo", "I am there yo")
        self.assertEqual(expected, result)
        expected = 0.5
        result = count_common_words("I am not here", "I am there yo")
        self.assertEqual(expected, result)
        expected = 0.5
        result = count_common_words("I am there yo", "I am not here")
        self.assertEqual(expected, result)
        expected = 0
        result = count_common_words("", "I am not here")
        self.assertEqual(expected, result)

