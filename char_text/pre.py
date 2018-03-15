#!/usr/bin/python
#-*-coding:utf-8 -*-
import os
import codecs


def get_median(data):
    data = sorted(data)
    size = len(data)
    if size % 2 == 0:  # 判断列表长度为偶数
        median = (data[size // 2] + data[size // 2 - 1]) / 2
        data[0] = median
    if size % 2 == 1:  # 判断列表长度为奇数
        median = data[(size - 1) // 2]
        data[0] = median
    return data[0]


def rand_file():
    input_file = os.path.join('data', 'train_cut.tsv')
    num = 0
    num1 = 0
    num2 = 0
    ori_list = []
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 3:
                continue
            num += 1
            ori_list.append(len(line[0].split()+line[1].split()))
            if num % 10000 == 0:
                print (num)
    print get_median(ori_list)
    print sum(ori_list)/600000.0
    print max(ori_list), min(ori_list)


if __name__ == "__main__":
    rand_file()
