from unittest import TestCase

from helpers.textual_analysis import *


class Test(TestCase):
    def test_count_common_words(self):
        expected = 2
        result = count_common_words("I am here", "I am there")
        self.assertEqual(expected, result)
        expected = 2
        result = count_common_words("I am not here", "I am there")
        self.assertEqual(expected, result)
        expected = 2
        result = count_common_words("I am there", "I am not here")
        self.assertEqual(expected, result)
        expected = 0
        result = count_common_words("", "I am not here")
        self.assertEqual(expected, result)

