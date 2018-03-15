#!/usr/bin/python
#-*-coding:utf-8 -*-

from gensim import corpora, models, similarities
import logging
import os
import codecs
import jieba
import re
import numpy as np
from zhon.hanzi import punctuation
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

str_stop = '的 了 和 是 就 都 而 及 與 著 或'
stop_list = set(str_stop.split())


def pad(s, fixed_length):
    ls = len(s)
    if ls >= fixed_length:
        s_pad = s[:fixed_length]
    else:
        seq_list = []
        num_repeat = fixed_length // ls
        for _ in range(num_repeat):
            seq_list.append(s)
        remain = fixed_length - num_repeat*ls
        if remain:
            seq_list.append(s[:remain])
        s_pad = ''.join(seq_list)
    return s_pad


def write_file(writer, seq_list, label):
    seq = []
    for st in range(32):
        seq_temp = seq_list[st].split()
        str_seq = ''.join(seq_temp)
        seq.append(str_seq)
    seq_join = ''.join(seq)
    if len(seq_join) != 1024:
        return False
    writer.write(label + '\t' + seq_join + '\n')
    return True


def process(s):
    l = ''.join([tk for tk in s.split() if tk])
    l = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), l)
    l = re.sub(ur"[%s]+" % punctuation, "", l)
    l = l.replace(' ', '')
    l = filter(lambda ch: ch not in ' \t1234567890:', l)
    l = filter(lambda c: c not in 'abcdefghijklmnopqrstuvwxyz', l)
    l = filter(lambda c: c not in 'ABCDEFGHIJKLMNOPQRSTUVMXYZ', l)
    return l


def gen():
    input_file = os.path.join('data', 'train.tsv')
    writer = codecs.open(os.path.join('data', 'train_tfidf_word.tsv'), 'w', encoding='utf-8')
    num = 0
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 4:
                continue
            num += 1
            print num
            seq_list = []
            title = process(line[1])
            content = process(line[2])

            seg_title = jieba.cut(title)
            seq_title = ' '.join(seg_title)

            if len(title):
                title_pad = pad(title, 32)
                seg = jieba.cut(title_pad)
                seq_list.append(' '.join(seg))

            ls = len(content)
            num_seq = ls // 32
            st = 0
            for _ in range(num_seq):
                seq = content[st:(st+32)]
                seg_list = jieba.cut(seq)
                seq_list.append(' '.join(seg_list))
                st += 32
            remain = ls - 32 * num_seq
            if remain:
                remain_pad = pad(content[:remain], 32)
                seg = jieba.cut(remain_pad)
                seq_list.append(' '.join(seg))

            texts = [[word for word in seq.lower().split() if word not in stop_list] for seq in seq_list]
            texts.append([word for word in seq_title.lower().split() if word not in stop_list])
            dictionary = corpora.Dictionary(texts)

            corpus = [dictionary.doc2bow(seq.lower().split()) for seq in seq_list]
            tfidf = models.TfidfModel(corpus)
            corpus_tfidf = tfidf[corpus]

            #lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
            #corpus_lsi = lsi[corpus_tfidf]

            query_bow = dictionary.doc2bow(seq_title.lower().split())
            query_tfidf = tfidf[query_bow]
            #query_lsi = lsi[query_tfidf]

            ls_corpus = len(corpus_tfidf)
            ls_query = len(query_tfidf)

            if not ls_corpus or not ls_query:
                new_seq_list = []
                if len(seq_list) >= 32:
                    for _id in range(32):
                        new_seq_list.append(seq_list[_id])
                else:
                    new_seq_list = seq_list
                    remain_ls = 32 - len(seq_list)
                    epoch = remain_ls // len(seq_list)
                    left = remain_ls - epoch * len(seq_list)
                    for _ in range(epoch):
                        new_seq_list.extend(seq_list)
                    for _id in range(left):
                        new_seq_list.append(seq_list[_id])
                write_file(writer, new_seq_list, line[0])

            if ls_corpus and ls_query:
                index = similarities.MatrixSimilarity(corpus_tfidf)
                sims = index[query_tfidf]
                sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
                new_seq_list = []
                id, sim = zip(*sort_sims)
                if len(seq_list) >= 32:
                    high_id = id[:16]
                    low_id = id[::-1][:16][::-1]
                    for _id in range(len(seq_list)):
                        if _id in high_id or _id in low_id:
                            new_seq_list.append(seq_list[_id])

                else:
                    new_seq_list = seq_list
                    remain_ls = 32 - len(seq_list)
                    epoch = remain_ls // len(seq_list)
                    left = remain_ls - epoch * len(seq_list)
                    for _ in range(epoch):
                        new_seq_list.extend(seq_list)
                    for _id in range(left):
                        new_seq_list.append(seq_list[_id])

                write_file(writer, new_seq_list, line[0])
            if num == 1000:
                break

    writer.close()

if __name__ == "__main__":
    gen()
