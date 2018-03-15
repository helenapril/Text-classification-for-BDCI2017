#!/usr/bin/python
#-*-coding:utf-8 -*-

import jieba
import codecs
import os
import collections
import cPickle
import numpy as np
import tensorflow as tf
import string
import re
import zhon
from zhon.hanzi import punctuation
import random


'''def creat_file():
    input_file = os.path.join('data1', 'train_tfidf.tsv')
    num = 0
    news_list = []
    writer = codecs.open(os.path.join('data', 'train_cut.tsv'), 'w', encoding='utf-8')
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 4:
                continue
            num += 1
            title = line[1]
            seg_title = jieba.cut(title)
            seq_title = ' '.join(seg_title)
            content = line[2]
            seg_content = jieba.cut(content)
            seq_content = ' '.join(seg_content)
            writer.write(seq_title[0] + '\t' + seq_content + '\t' + line[3] + '\n')
            if num % 10000 == 0:
                print (num)


num = 0
input_file = os.path.join('data', 'eval_tfidf.tsv')
with codecs.open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        line = line.split('\t')
        if len(line) != 2:
            continue
        seq = line[1]
        if len(seq) != 1024:
            continue
        num += 1
        if num % 10000 == 0:
            print (num)
    print num



line = u"厨亮灶”。 中午11时许，"
l = ''.join([tk for tk in line.split() if tk])
l = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),l)
l = re.sub(ur"[%s]+" %punctuation, "", l) # 需要将str转换为unicode
l = l.replace(' ', '')
l = filter(lambda ch: ch not in ' \t1234567890', l)
l = filter(lambda c: c not in 'abcdefghijklmnopqrstuvwxyz', l)
l = filter(lambda c: c not in 'ABCDEFGHIJKLMNOPQRSTUVMXYZ', l)
print l'''

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

input_data = tf.Variable([[[0.2, 0.1, 0.2],
                           [0.1, 0.9, 0],
                           [0.5, 0.5, 0]],
                          [[0.2, 0.1, 0.2],
                           [0.1, 0.9, 0],
                           [0.5, 0.5, 0]]], dtype=tf.float32)
output = tf.nn.softmax(input_data)
output2 = tf.nn.softmax(input_data)
output1 = tf.reverse_sequence(output2, seq_lengths=[2, 2], seq_axis=1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(output2)
    print sess.run(output1)
    '''list_logits = tf.split(output, num_or_size_splits=2, axis=1)
    logit = tf.squeeze(list_logits[1])
    print sess.run(logit)
    ans = tf.greater_equal(logit, tf.constant(0.6))
    res = tf.cast(ans, dtype=tf.int64)
    print sess.run(res)
    res = []
    input_data = sess.run(input_data)
    for data in input_data:
        data = softmax(data)
        print data
        if data[1] >= 0.6:
            res.append(1)
        else:
            res.append(0)
    print(res)'''




