#!/usr/bin/python
#-*-coding:utf-8 -*-
import os
import codecs
import jieba
import re
from zhon.hanzi import punctuation
import collections
import cPickle
import gensim
from gensim.models.word2vec import Word2Vec
import multiprocessing
import numpy as np


def process(s):
    l = ''.join([tk for tk in s.split() if tk])
    l = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"), l)
    l = re.sub(ur"[%s]+" % punctuation, "", l)
    l = l.replace(' ', '')
    l = filter(lambda ch: ch not in ' \t1234567890:', l)
    l = filter(lambda c: c not in 'abcdefghijklmnopqrstuvwxyz', l)
    l = filter(lambda c: c not in 'ABCDEFGHIJKLMNOPQRSTUVMXYZ', l)
    return l


def write_file(writer, title, content, label):
    writer.write(title + '\t' + content + '\t' + label + '\n')


def gen():
    input_file = os.path.join('data', 'train.tsv')
    writer = codecs.open(os.path.join('data', 'train_cut.tsv'), 'w', encoding='utf-8')
    num = 0
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 4:
                continue
            num += 1
            title = process(line[1])
            content = process(line[2])

            seg_title = jieba.cut(title)
            seq_title = ' '.join(seg_title)
            seg_content = jieba.cut(content)
            seq_content = ' '.join(seg_content)
            write_file(writer, seq_title, seq_content, line[3])
            if num % 1000 == 0:
                print num


def vocab():
    data = []
    input_file = os.path.join('data', 'train_cut.tsv')
    num = 0
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 3:
                continue
            num += 1
            title = line[0].split()
            content = line[1].split()
            data.extend(title)
            data.extend(content)
            if num == 100:
                print (num)
                break

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, _ = zip(*count_pairs)
    print (len(chars))
    chars = chars[:50000]
    vocab = dict(zip(chars, range(1, len(chars) + 1)))
    vocab['unk'] = len(chars) + 1
    vocab['pad'] = 0
    vocab_size = len(vocab)
    print ('size of char vocabulary:', vocab_size)
    with open(os.path.join('data', 'word_vocab.pkl'), 'wb') as f:
        cPickle.dump(vocab, f)


def construct():
    input_file = os.path.join('data', 'train_cut.tsv')
    writer = codecs.open(os.path.join('data', 'train_gensim.tsv'), 'wb', encoding='utf-8')
    vocab_file = os.path.join('data', 'word_vocab.pkl')
    num = 0
    with open(vocab_file, 'rb') as f:
        word_vocab = cPickle.load(f)
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            num += 1
            data = []
            if len(line) == 2:
                content = line[0].split()
                for word in content:
                    if word not in word_vocab:
                        data.append('unk')
                    else:
                        data.append(word)
            if len(line) == 3:
                title = line[0].split()
                content = line[1].split()
                for word in title:
                    if word not in word_vocab:
                        data.append('unk')
                    else:
                        data.append(word)
                for word in content:
                    if word not in word_vocab:
                        data.append('unk')
                    else:
                        data.append(word)
            seq = ' '.join(data)
            writer.write(seq + '\n')
            if num % 1000 == 0:
                print num


def cons():
    input_file = os.path.join('data', 'train_gensim.tsv')
    writer = codecs.open(os.path.join('data', 'train_gensim_cut.tsv'), 'wb', encoding='utf-8')
    num = 0
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            ls = len(line)
            if not ls:
                continue
            num += 1
            num_seq = ls / 15
            remain = ls - num_seq * 15
            st = 0
            for _ in range(num_seq):
                seq = line[st: st+15]
                st += 15
                writer.write(' '.join(seq) + '\n')
            if remain:
                seq = line[st:ls]
                writer.write(' '.join(seq) + '\n')
            if num % 10000 == 0:
                print num


def word2vec_train():
    input_file = os.path.join('data', 'train_gensim.tsv')
    sentences = gensim.models.word2vec.LineSentence(input_file)
    model = gensim.models.Word2Vec(sentences, size=128, window=5, min_count=10, sg=0, workers=multiprocessing.cpu_count())
    model.save_word2vec_format('data/w2v_model.save' + '.vector', binary=True)
    model.save('data/w2v_model.save')


def test_word():
    model = Word2Vec.load('data/w2v_model.save')
    word_vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    #word_vocab_reverse = dict([(v.index, k) for k, v in model.wv.vocab.items()])
    #print word_vocab_reverse[1]
    print len(word_vocab)
    li = model.most_similar([u'项羽'], topn =10)
    for l in li:
        print l[0]
    with open(os.path.join('data', 'word_to_vec_vocab.pkl'), 'rb') as f:
        word_vocab = cPickle.load(f)
    print word_vocab[u'数据']
    weights = model.wv.syn0
    embedding = np.array(weights)
    print embedding[word_vocab[u'数据']]
    print model[u'数据']

if __name__ == "__main__":
    #gen()
    #vocab()
    #cons()
    #word2vec_train()
    test_word()