from unittest import TestCase
from helpers.data_preprocessing import DataProcesser
import pandas as pd
import numpy as np


class TestDataProcesser(TestCase):
    def test_remove_java_tags(self):
        dp = DataProcesser()
        # test basic tag removal
        text = '@author this'
        expected = 'this'
        result = dp.remove_java_tags(text).lstrip()
        self.assertEqual(expected, result)
        # test tag removal based on regexp
        text = '{@link szdsdzsdz} this'
        expected = 'this'
        result = dp.remove_java_tags(text).lstrip()
        self.assertEqual(expected, result)
        # test no tag removal
        text = 'as this'
        expected = 'as this'
        result = dp.remove_java_tags(text).lstrip()
        self.assertEqual(expected, result)

    def test_preprocess(self):
        dp = DataProcesser()
        data = '@Override this'
        result = dp.preprocess(data)
        self.assertEqual('', result)

    def test_extract_camel_case(self):
        dp = DataProcesser()
        expected = 'Camel Case'
        received = dp.extract_camel_case('CamelCase')
        self.assertEqual(expected, received)
        # test multiple camel case
        expected = 'Camel Case Camel Case'
        received = dp.extract_camel_case('CamelCaseCamelCase')
        self.assertEqual(expected, received)
        # test non-camel case
        expected = 'Camel Case'
        received = dp.extract_camel_case('Camel Case')
        self.assertEqual(expected, received)


    def test_extract_snake_case(self):
        dp = DataProcesser()
        expected = 'snake case'
        received = dp.extract_snake_case('snake_case')
        self.assertEqual(expected, received)
        # test multiple snake case
        expected = 'snake case oh yea'
        received = dp.extract_snake_case('snake_case_oh_yea')
        self.assertEqual(expected, received)
