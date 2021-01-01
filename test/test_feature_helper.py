from unittest import TestCase
from helpers.feature_helper import FeatureHelper


class TestFeatureHelper(TestCase):
    def test_get_java_tags_num(self):
        string = "@implNote taken from {@link com.sun.javafx.scene.control.behavior.TextAreaBehavior#contextMenuRequested(javafx.scene.input.ContextMenuEvent)}"
        self.assertEqual(2, FeatureHelper.get_java_tags_num(string))
        string = "@implNote @hello"
        self.assertEqual(1, FeatureHelper.get_java_tags_num(string))
        string = ""
        self.assertEqual(0, FeatureHelper.get_java_tags_num(string))
