#!/usr/bin/python
#-*-coding:utf-8 -*-

import codecs
import os
import cPickle
import numpy as np
import tensorflow as tf


def make_example(sequence, labels):

    input_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[token]))
        for token in sequence]
    label_features = [
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        for label in labels]
    feature_list = {
        'inputs': tf.train.FeatureList(feature=input_features),
        'labels': tf.train.FeatureList(feature=label_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    return tf.train.SequenceExample(feature_lists=feature_lists)


def sample(input_file):
    num = 0
    num_pos = 0
    res_pos = []
    num_neg = 0
    res_neg = []
    input_file = os.path.join('data', input_file)
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 2:
                continue
            num += 1
            if num_pos == 180000 and num_neg == 180000:
                break
            if line[1] == 'POSITIVE':
                if num_pos == 180000:
                    continue
                num_pos += 1
                res_pos.append([line[0], line[1]])
            if line[1] == 'NEGATIVE':
                if num_neg == 180000:
                    continue
                num_neg += 1
                seq = []
                res_neg.append([line[0], line[1]])
            if num % 10000 == 0:
                print num

    num = 0
    output_file = os.path.join('data', 'train_tfifd_sample.tsv')
    with codecs.open(output_file, 'w', encoding='utf-8') as writer:
        print ('write:')
        for pos, neg in zip(res_pos, res_neg):
            num += 1
            writer.write(pos[0] + '\t' + pos[1] + '\n')
            writer.write(neg[0] + '\t' + neg[1] + '\n')
            if num % 10000 == 0:
                print (num)


class TextLoader():
    def __init__(self, filename):

        input_file = os.path.join('data', filename)
        vocab_file = os.path.join('data', "char_vocab.pkl")
        file_1 = os.path.join('data', 'valid_tfidf_tfrecord')
        self.preprocess(input_file, vocab_file, file_1)

    def preprocess(self, input_file, vocab_file, write_to_file_1):
        with open(vocab_file, 'rb') as f:
            vocab = cPickle.load(f)
        self.vocab_size = len(vocab)
        print (self.vocab_size)
        num = 0
        writer_1 = tf.python_io.TFRecordWriter(write_to_file_1)
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line.split('\t')
                if len(line) != 2:
                    continue
                content = line[0]
                label = line[1]
                if not content:
                    continue
                num += 1
                sequence = [vocab[char] if char in vocab else vocab['unk'] for char in content]
                sequence = np.array(sequence)
                if label == 'POSITIVE':
                    label_sequence = np.array([0, 1])
                else:
                    label_sequence = np.array([1, 0])
                ex = make_example(sequence, label_sequence)
                writer_1.write(ex.SerializeToString())
                if num % 1000 == 0:
                    print num
        writer_1.close()
        print (num, self.vocab_size)
        print("Wrote to {}".format(write_to_file_1))


if __name__ == "__main__":
    loader = TextLoader('valid_tfidf.tsv')

