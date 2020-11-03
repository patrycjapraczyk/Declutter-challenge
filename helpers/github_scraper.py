import requests
import bs4 as bs
import pandas as pd
import numpy as np


import time

LINE_COMMENT_KEY = '//'
MULTIPLE_LINE_COMMENT_START = '*/'


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


def get_preceding_code_line_comment(comment_line: str) -> str:
    comment_start = comment_line.find(LINE_COMMENT_KEY)
    preceding_code = comment_line
    if comment_start != -1:
        preceding_code = comment_line[:comment_start]
    return preceding_code.strip()


def get_code_line_comment(path: str, begin_line: int) -> str:
    line = get_code_line(path, begin_line)
    counter = 1
    if not is_line_comment(line):
        return get_preceding_code_line_comment(line)

    while line.isspace() or is_line_comment(line):
        line = get_code_line(path, begin_line + counter)
        counter += 1
    return line


def handle_block_comment(path, begin_line):
    line = get_code_line(path, begin_line)
    counter = 1
    found_comment_end = False
    while not found_comment_end or line.isspace():
        if MULTIPLE_LINE_COMMENT_START in line:
            found_comment_end = True
        line = get_code_line(path, begin_line + counter)
        counter += 1
    return line


data = pd.read_csv("./../data/train_set_0520.csv", usecols=['type', 'path_to_file', 'begin_line'])
paths = data['path_to_file'].tolist()
paths = paths[560:]
begin_lines = data['begin_line'].tolist()
begin_lines = begin_lines[560:]
types = data['type'].tolist()
types = types[560:]

lines = []

for i in range(0, len(paths)):
    print('-------')
    print(paths[i])
    print(begin_lines[i])
    line = ""
    if types[i] == 'Line':
        line = get_code_line_comment(paths[i], begin_lines[i] + 1)
    elif types[i] == 'Block':
        line = handle_block_comment(paths[i], begin_lines[i])
    else:
        line = handle_block_comment(paths[i], begin_lines[i] + 1)
    print(line)
    lines.append(line)
    f = open("./../data/code2.csv", "a")
    f.write(line + '\n')
    f.close()

df2 = pd.DataFrame(np.array(lines),
                   columns=['comments'])
df2.to_csv('./../data/comments.csv')

