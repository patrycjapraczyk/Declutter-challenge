import requests
import bs4 as bs
import pandas as pd
import numpy as np

from const.java_tags import *
from const.programming_keywords import *

import time
import re
from helpers.data_preprocessing import DataProcesser

LINE_COMMENT_KEY = '//'
MULTIPLE_LINE_COMMENT_START = '/*'
MULTIPLE_LINE_COMMENT_END = '*/'
PARENS_OPEN = '{'
PARENS_CLOSE = '}'


def do_request(link: str):
    """
    performs get request on a link- returns the entire HTML code of a page
    """
    val = requests.get(link).text
    time.sleep(5)
    return val


def get_code_line(link: str, line_num: int, result=None) -> str:
    """
    gets html from @link and returns a string
    with the contents at @line_num
    :param link: a link to file on GitHub repository
    :param line_num: a code line
    :return: file content at @line_num
    """
    if result == None:
        result = requests.get(link).text
        time.sleep(5)
    soup = bs.BeautifulSoup(result, 'lxml')
    # comment lines normally have LC+line_number html id tag
    line_id = "LC" + str(line_num)
    tag = soup.find(id=line_id)
    line_text = ""
    if tag:
        children = tag.children
        for child in children:
            if hasattr(child, 'children'):
                line_text += child.text
            else:
                line_text += child.string

    return line_text.strip()


def is_line_comment(code_line: str) -> bool:
    code_line = code_line.lstrip()
    return code_line[:2] == LINE_COMMENT_KEY


def is_block_comment(code_line: str) -> bool:
    code_line = code_line.lstrip()
    return code_line[:2] == MULTIPLE_LINE_COMMENT_START


def get_preceding_code_comment(comment_line: str, line_comment=True) -> str:
    """
        returns code preceding a comment on the same line if there is such code,
        otherwise returns an empty string ''
    """
    comment_key = LINE_COMMENT_KEY
    if not line_comment:
        comment_key = MULTIPLE_LINE_COMMENT_START
    comment_start = comment_line.find(comment_key)
    preceding_code = comment_line
    if comment_start != -1:
        preceding_code = comment_line[:comment_start]
    return preceding_code.strip()


def is_line_java_tag(code_line: str) -> bool:
    """
    returns True if the comment contains only a Java tag
    """
    for tag in JAVA_TAGS:
        if re.match(rf"{tag}", code_line, re.IGNORECASE):
            return True
    return False


def is_line_java_keyword(code_line: str) -> bool:
    """
       returns True if the comment contains only a Java keyword
    """
    code_line = DataProcesser.remove_special_characters(code_line)
    for keyword in JAVA_KEYWORDS:
        if code_line == keyword:
            return True
    return False


def is_only_special_char(code_line: str) -> str:
    code_line = code_line.replace(" ", ",")
    code_line = DataProcesser.remove_special_characters(code_line)
    return code_line == ''


def is_invalid_code(code_line: str) -> bool:
    """
    returns True if the code line is empty, is a comment,
    contains only a Java tag, only a Java keyword or only a special character
    """
    is_com = is_line_comment(code_line)
    is_tag = is_line_java_tag(code_line)
    code_line = DataProcesser.remove_special_characters(code_line)
    code_line = code_line.replace(" ", "")
    return (code_line.isspace() or is_com or \
            is_tag or is_line_java_keyword(code_line) or is_only_special_char(code_line))


def get_code_line_comment(path: str, begin_line: int) -> str:
    code_line = get_code_line(path, begin_line)
    counter = 1
    if not is_line_comment(code_line):
        return get_preceding_code_comment(code_line)

    while is_invalid_code(code_line) and not("}" in code_line):
        code_line = get_code_line(path, begin_line + counter)
        counter += 1

    counter = -2
    if is_only_special_char(code_line):
        code_line = get_code_line(path, begin_line + counter)
        if is_invalid_code(code_line):
            code_line = ""

    return code_line


def handle_block_comment(path, begin_line):
    code_line = get_code_line(path, begin_line)
    found_comment_end = False
    if MULTIPLE_LINE_COMMENT_END in code_line:
        found_comment_end = True

    if not is_block_comment(code_line):
        return get_preceding_code_comment(code_line, False)
    else:
        code_line = get_code_line(path, begin_line + 1)

    counter = 1
    while not found_comment_end:
        if MULTIPLE_LINE_COMMENT_END in code_line:
            found_comment_end = True
        code_line = get_code_line(path, begin_line + counter)
        counter += 1

    while is_invalid_code(code_line) and not("}" in code_line):
        code_line = get_code_line(path, begin_line + counter)
        counter += 1

    if is_only_special_char(code_line):
        if is_invalid_code(code_line):
            code_line = ""

    return code_line


def handle_javadoc_code(path, begin_line):
    result = do_request(path)
    code_line = get_code_line(path, begin_line, result)
    counter = 1
    found_comment_end = False
    while not found_comment_end:
        if MULTIPLE_LINE_COMMENT_END in code_line:
            found_comment_end = True
        code_line = get_code_line(path, begin_line + counter, result)
        counter += 1

    while is_invalid_code(code_line) and not ("}" in code_line):
        code_line = get_code_line(path, begin_line + counter, result)
        counter += 1

    parens_open = 0
    total_code = code_line
    is_first = True
    while parens_open > 0 or is_first:
        is_first = False
        count_open_parens = code_line.count(PARENS_OPEN)
        count_parens_close = code_line.count(PARENS_CLOSE)
        parens_open += count_open_parens - count_parens_close
        total_code += " " + code_line
        code_line = get_code_line(path, begin_line + counter, result)
        counter += 1

    return total_code


def collect_javadoc_comments():
    data = pd.read_csv("./../data/train_set_0520.csv", usecols=['type', 'path_to_file', 'begin_line'])
    paths = data['path_to_file'].tolist()
    begin_lines = data['begin_line'].tolist()
    types = data['type'].tolist()

    lines = []

    for i in range(0, len(paths)):
        print('-------')
        print(paths[i])
        print(begin_lines[i])
        code = ""
        if types[i] == 'Javadoc':
            code = handle_javadoc_code(paths[i], begin_lines[i])

        print(code)
        lines.append(code)
        f = open("./../data/javadoc_new_code.csv", "a")
        f.write(code + '\n')
        f.close()


# collects code for Javadoc comments
collect_javadoc_comments()
