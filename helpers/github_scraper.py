import requests
import bs4 as bs
import pandas as pd
import numpy as np

from const.java_tags import *
from const.java_keywords import *

import time
import re
from helpers.data_preprocessing import DataProcesser

LINE_COMMENT_KEY = '//'
MULTIPLE_LINE_COMMENT_START = '/*'
MULTIPLE_LINE_COMMENT_END = '*/'


def get_code_line(link: str, line_num: int) -> str:
    """
    gets html from @link and returns a string
    with the contents at @line_num
    :param link: a link to file on GitHub repository
    :param line_num: a code line
    :return: file content at @line_num
    """
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
    comment_key = LINE_COMMENT_KEY
    if not line_comment:
        comment_key = MULTIPLE_LINE_COMMENT_START
    comment_start = comment_line.find(comment_key)
    preceding_code = comment_line
    if comment_start != -1:
        preceding_code = comment_line[:comment_start]
    return preceding_code.strip()


def is_line_java_tag(code_line: str) -> bool:
    for tag in JAVA_TAGS:
        if re.match(rf"{tag}", code_line, re.IGNORECASE):
            return True
    return False


def is_line_java_keyword(code_line: str) -> bool:
    dp = DataProcesser()
    code_line = dp.remove_special_characters(code_line)
    for keyword in JAVA_KEYWORDS:
        if code_line == keyword:
            return True
    return False


def is_only_special_char(code_line: str) -> str:
    dp = DataProcesser()
    code_line = dp.remove_special_characters(code_line)
    code_line = code_line.replace(" ", "")
    return code_line == ''


def is_invalid_code(code_line: str) -> bool:
    is_com = is_line_comment(code_line)
    is_tag = is_line_java_tag(code_line)
    dp = DataProcesser()
    code_line = dp.remove_special_characters(code_line)
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
    if not is_block_comment(code_line):
        return get_preceding_code_comment(code_line, False)
    else:
        code_line = get_code_line(path, begin_line + 1)

    counter = 1
    found_comment_end = False
    while not found_comment_end:
        if MULTIPLE_LINE_COMMENT_END in code_line:
            found_comment_end = True
        code_line = get_code_line(path, begin_line + counter)
        counter += 1

    while is_invalid_code(code_line) and not("}" in code_line):
        code_line = get_code_line(path, begin_line + counter)
        counter += 1

    counter = -1
    if is_only_special_char(code_line):
        code_line = get_code_line(path, begin_line + counter)
        if is_invalid_code(code_line):
            code_line = ""

    return code_line


data = pd.read_csv("./../data/train_set_0520.csv", usecols=['type', 'path_to_file', 'begin_line'])
paths = data['path_to_file'].tolist()[1311:] #206 559 560 1312
begin_lines = data['begin_line'].tolist()[1311:]
types = data['type'].tolist()[1311:]

lines = []

for i in range(0, len(paths)):
    print('-------')
    print(paths[i])
    print(begin_lines[i])
    line = ""
    if types[i] == 'Line':
        line = get_code_line_comment(paths[i], begin_lines[i])
    elif types[i] == 'Block':
        line = handle_block_comment(paths[i], begin_lines[i])
    else:
        line = handle_block_comment(paths[i], begin_lines[i])
    print(line)
    lines.append(line)
    f = open("./../data/code_data.csv", "a")
    f.write(line + '\n')
    f.close()


df2 = pd.DataFrame(np.array(lines),
                   columns=['comments'])
df2.to_csv('./../data/comments.csv')
