import random
import os
import codecs
import tensorflow as tf
import cPickle
import numpy as np


def rand_file():
    input_file = os.path.join('data', 'train_gensim_cut.tsv')
    writer1 = codecs.open(os.path.join('data', 'train_rand_gensim_cut.tsv'), 'w', encoding='utf-8')
    writer2 = codecs.open(os.path.join('data', 'valid_gensim_cut.tsv'), 'w', encoding='utf-8')
    num = 0
    ori_list = []
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            num += 1
            ori_list.append(line)

    random_list = random.sample(range(600000), 600000)
    print (len(random_list))

    count = 0
    for num in random_list:
        count += 1
        line = ori_list[num]
        if count % 1000 == 0:
            print (count)
        if count <= 595000:
            writer1.write(line)
        else:
            writer2.write(line)


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


class TextLoader():
    def __init__(self, filename):

        input_file = os.path.join('data', filename)
        input_file2 = os.path.join('data', 'valid_tfidf.tsv')
        vocab_file = os.path.join('data', "char_vocab.pkl")
        file_1 = os.path.join('data', 'train_tfidf_rand_tfrecord')
        file_2 = os.path.join('data', 'valid_tfidf_tfrecord')
        self.preprocess(input_file, vocab_file, file_1)
        self.preprocess(input_file2, vocab_file, file_2)

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
                if not content:
                    continue
                num += 1
                label = line[1]
                sequence = []
                content = content[::-1][:1024][::-1]
                if len(content) >= 1024:
                    sequence = [vocab[char] if char in vocab else vocab['unk'] for char in content]
                else:
                    epoch = 1024 // len(content)
                    for _ in range(epoch):
                        for char in content:
                            if char in vocab:
                                sequence.append(vocab[char])
                            else:
                                sequence.append(vocab['unk'])
                    remain = 1024 - (epoch * len(content))
                    if remain:
                        for st in range(remain):
                            if content[st] in vocab:
                                sequence.append(vocab[content[st]])
                            else:
                                sequence.append(vocab['unk'])

                sequence = np.array(sequence)
                if label == 'POSITIVE':
                    label_sequence = np.array([0, 1])
                else:
                    label_sequence = np.array([1, 0])
                ex = make_example(sequence, label_sequence)
                writer_1.write(ex.SerializeToString())
                if num % 100 == 0:
                    print (num)
        writer_1.close()
        print (num, self.vocab_size)
        print("Wrote to {}".format(write_to_file_1))


if __name__ == "__main__":
    #rand_file()
    loader = TextLoader('train_tfidf_rand.tsv')
