from unittest import TestCase
from helpers.github_scraper import get_code_line, is_line_comment, get_preceding_code_comment, get_code_line_comment, is_only_special_char, handle_block_comment, is_line_java_keyword, is_line_java_tag


class Test(TestCase):
    def test_get_code_line(self):
        # test comment
        result1 = get_code_line(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/desktop/JabRefDesktop.java",
            83)
        expected1 = "// should be opened in browser"
        self.assertEqual(expected1, result1)
        # test empty
        result2 = get_code_line(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/texparser/CitationsDisplay.java",
            60)
        expected2 = ""
        self.assertEqual(expected2, result2)
        # test code
        result3 = get_code_line(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/importer/ParserResultWarningDialog.java",
            26)
        expected3 = "Objects.requireNonNull(parserResult);"
        self.assertEqual(expected3, result3)
        # test first line
        result4 = get_code_line(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/util/CustomLocalDragboard.java",
            1)
        expected4 = "package org.jabref.gui.util;"
        self.assertEqual(expected4, result4)
        # test last line
        result5 = get_code_line(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/importer/actions/AppendDatabaseAction.java",
            189)
        expected5 = "}"
        self.assertEqual(expected5, result5)
        # test non-existent line
        result6 = get_code_line(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/importer/actions/AppendDatabaseAction.java",
            199)
        expected6 = ""
        self.assertEqual(expected6, result6)

    def test_is_line_comment(self):
        self.assertEqual(True, is_line_comment("//"))
        self.assertEqual(True, is_line_comment("//ejwkehwkjeh"))
        self.assertEqual(True, is_line_comment('// jkwhjkr jehjkrh jrehjrkh jerhjkerh kejrhekjh //'))
        self.assertEqual(False, is_line_comment("eeeeeeawe waewae "))
        self.assertEqual(False, is_line_comment("kjkldfj // dsfkjfksd"))
        self.assertEqual(True, is_line_comment("// Always overwrite keys by default"))

    def test_get_preceding_code_line_comment(self):
        result = get_preceding_code_comment("code here // comment here")
        expected = "code here"
        self.assertEqual(expected, result)
        # test no preceding code
        result = get_preceding_code_comment("//comment")
        self.assertEqual("", result)
        # test empty string
        result = get_preceding_code_comment("")
        self.assertEqual("", result)
        #test no code
        result = get_preceding_code_comment("comment")
        self.assertEqual("comment", result)
        # test block comment
        result = get_preceding_code_comment("com p/*daskdjk ksaldj", False)
        self.assertEqual("com p", result)

    def test_get_code_line_comment(self):
        result = get_code_line_comment("https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/Globals.java", 40)
        expected = "public static final RemoteListenerServerLifecycle REMOTE_LISTENER = new RemoteListenerServerLifecycle();"
        self.assertEqual(expected, result)
        result = get_code_line_comment("https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/externalfiletype/ExternalFileTypes.java", 228)
        expected = "ExternalFileType toRemove = null;"
        self.assertEqual(expected, result)
        result = get_code_line_comment("https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/bibtexkeypattern/GenerateBibtexKeyAction.java", 51)
        expected = "return true;"
        self.assertEqual(expected, result)
        result = get_code_line_comment("https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/exporter/ExportToClipboardAction.java", 90)
        self.assertEqual("tmp = Files.createTempFile(\"jabrefCb\", \".tmp\");", result)

    def test_handle_block_comment(self):
        result = handle_block_comment("https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/util/BindingsHelper.java", 54)
        expected = "public static <A, B> MappedList<B, A> mapBacked(ObservableList<A> source, Function<A, B> mapper) {"
        self.assertEqual(expected, result)
        result = handle_block_comment("https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/model/TreeNode.java", 47)
        self.assertEqual("private Consumer<T> onDescendantChanged = t -> {", result)
        result = handle_block_comment(
            "https://github.com/nnovielli/jabref/blob/master/src/main/java/org/jabref/gui/actions/OldCommandWrapperForActiveDatabase.java",
            12)
        expected = "public class OldCommandWrapperForActiveDatabase extends CommandBase {"
        self.assertEqual(expected, result)

    def test_is_line_java_tags(self):
        result = is_line_java_tag("@Override")
        self.assertEqual(True, result)
        result = is_line_java_tag("//djd")
        self.assertEqual(False, result)
        result = is_line_java_tag("skjf djfhjdf jdfhjfds")
        self.assertEqual(False, result)
        result = is_line_java_tag("{@link ww.jsdkh.com}")
        self.assertEqual(True, result)

    def test_is_line_java_keyword(self):
        result = is_line_java_keyword("else")
        self.assertEqual(True, result)
        result = is_line_java_keyword("}else{")
        self.assertEqual(True, result)
        result = is_line_java_keyword("}switch{")
        self.assertEqual(True, result)

    def test_is_only_special_char(self):
        result = is_only_special_char("")
        self.assertEqual(True, result)
        result = is_only_special_char("?}")
        self.assertEqual(True, result)
        self.assertEqual(True, is_only_special_char(" "))
        self.assertEqual(False, is_only_special_char("hello }d."))


