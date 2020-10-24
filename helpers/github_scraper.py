import requests
import bs4 as bs
import pandas as pd
import numpy as np


import time

LINE_COMMENT_KEY = '//'

def get_code_line(link, line_num):
    result = requests.get(link).text
    time.sleep(5)
    soup = bs.BeautifulSoup(result, 'lxml')
    line_id = "LC" + str(line_num)
    tag = soup.find(id=line_id)
    #code_line = tag.findChild(class_="pl-c")
    line_text = ""
    if tag:
        children = tag.children
        for child in children:
            if hasattr(child, 'children'):
                line_text += child.text
            else:
                line_text += child.string

    return line_text


def is_comment(line: str):
    line = line.lstrip()
    return line[:2] == LINE_COMMENT_KEY


def handle_line_comment(path, begin_line):
    line = get_code_line(path, begin_line)
    counter = 1
    while line.isspace() or is_comment(line):
        line = get_code_line(path, begin_line + counter)
        counter += 1
    return line


def handle_block_comment(path, begin_line):
    line = get_code_line(path, begin_line)
    counter = 1
    found_comment_end = False
    while not found_comment_end or line.isspace():
        if '*/' in line:
            found_comment_end = True
        line = get_code_line(path, begin_line + counter)
        counter += 1
    return line


data = pd.read_csv("./../data/train_set_0520.csv", usecols=['type', 'path_to_file', 'begin_line'])
paths = data['path_to_file'].tolist()
begin_lines = data['begin_line'].tolist()
types = data['type'].tolist()

lines = []

for i in range(0, len(paths)):
    print('-------')
    print(paths[i])
    print(begin_lines[i])
    line = ""
    if types[i] == 'Line':
        line = handle_line_comment(paths[i], begin_lines[i] + 1)
    elif types[i] == 'Block':
        line = handle_block_comment(paths[i], begin_lines[i] + 1)
    else:
        line = handle_block_comment(paths[i], begin_lines[i] + 1)
    print(line)
    lines.append(line)
    f = open("./../data/comments_test.csv", "a")
    f.write(line + '\n')
    f.close()

df2 = pd.DataFrame(np.array(lines),
                   columns=['comments'])
df2.to_csv('./../data/comments.csv')

