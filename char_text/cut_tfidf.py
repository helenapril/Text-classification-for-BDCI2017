import random
import os
import codecs
import tensorflow as tf
import cPickle
import numpy as np


def rand_file():
    input_file = os.path.join('data', 'train_cut.tsv')
    writer1 = codecs.open(os.path.join('data', 'train_rand_cut.tsv'), 'w', encoding='utf-8')
    writer2 = codecs.open(os.path.join('data', 'valid_cut.tsv'), 'w', encoding='utf-8')
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


def pad(s, fixed_length):
    ls = len(s)
    if ls >= fixed_length:
        s_pad = s[:fixed_length]
        s_pad = ' '.join(s_pad)
    else:
        seq_list = []
        num_repeat = fixed_length // ls
        for _ in range(num_repeat):
            seq_list.extend(s)
        remain = fixed_length - num_repeat*ls
        if remain:
            seq_list.extend(s[:remain])
        s_pad = ' '.join(seq_list)
    return s_pad


def write_to_file():
    input_file = os.path.join('data', 'eval_cut.tsv')
    writer = codecs.open(os.path.join('data', 'eval_512_cut.tsv'), 'w', encoding='utf-8')
    num = 0
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            new_line = process(line)
            writer.write(new_line + '\n')
            num += 1
            if num % 10000 == 0:
                print num


def process(line):
    line = line.strip()
    line = line.split('\t')
    new_content_list = []
    if len(line) == 3:
        title = line[1].split()
        content = line[2].split()
        if len(title) == 0:
            ls_content = len(content)
            num_seq = ls_content // 16
            content_list = []
            st = 0
            for _ in range(num_seq):
                seg = content[st: st + 16]
                st += 16
                content_list.append(' '.join(seg))
            remain = ls_content - num_seq * 16
            if remain:
                seg = pad(content[st:ls_content], 16)
                content_list.append(seg)
            if len(content_list) >= 32:
                for id in range(32):
                    new_content_list.append(content_list[id])
            else:
                num_list = 32 // len(content_list)
                remain = 32 - len(content_list) * num_list
                for _ in range(num_list):
                    for seq in content_list:
                        new_content_list.append(seq)
                for id in range(remain):
                    new_content_list.append(content_list[id])

            new_line = line[0] + '\t' + ' '.join(new_content_list)

        else:
            new_title = pad(title, 16)
            new_content_list.append(new_title)
            ls_content = len(content)
            num_seq = ls_content // 16
            content_list = []
            st = 0
            for _ in range(num_seq):
                seg = content[st: st+16]
                st += 16
                content_list.append(' '.join(seg))
            remain = ls_content - num_seq*16
            if remain:
                seg = pad(content[st:ls_content], 16)
                content_list.append(seg)
            if len(content_list) >= 31:
                for id in range(31):
                    new_content_list.append(content_list[id])
            elif len(content_list) == 0:
                for _ in range(31):
                    new_content_list.append(new_title)
            else:
                num_list = 31//len(content_list)
                remain = 31 - len(content_list) * num_list
                for _ in range(num_list):
                    for seq in content_list:
                        new_content_list.append(seq)
                for id in range(remain):
                    new_content_list.append(content_list[id])

            new_line = line[0] + '\t' + ' '.join(new_content_list)

    if len(line) == 2:
        content = line[1].split()
        ls_content = len(content)
        num_seq = ls_content // 16
        content_list = []
        st = 0
        for _ in range(num_seq):
            seg = content[st: st + 16]
            st += 16
            content_list.append(' '.join(seg))
        remain = ls_content - num_seq * 16
        if remain:
            seg = pad(content[st:ls_content], 16)
            content_list.append(seg)
        if len(content_list) >= 32:
            new_content_list = content_list[:32]
        else:
            num_list = 32 // len(content_list)
            remain = 32 - len(content_list) * num_list
            for _ in range(num_list):
                for seq in content_list:
                    new_content_list.append(seq)
            for id in range(remain):
                new_content_list.append(content_list[id])

        new_line = line[0] + '\t' + ' '.join(new_content_list)
    return new_line


def check():
    input_file = os.path.join('data', 'eval_512_cut.tsv')
    num = 0
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 2:
                continue
            seq = line[1].split()
            if len(seq) != 512:
                continue
            num += 1
            if num % 100 == 0:
                print (num)
        print num


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
        vocab_file = os.path.join('data', "word_to_vec_vocab.pkl")
        file_1 = os.path.join('data', 'train_512_tfrecord')
        file_2 = os.path.join('data', 'valid_512_tfrecord')
        file_3 = os.path.join('data', 'eval_512_tfrecord')
        if filename == 'train':
            files = [file_1]
            input_file = os.path.join('data', 'train_512_rand_cut.tsv')
        if filename == 'valid':
            files = [file_2]
            input_file = os.path.join('data', 'valid_512_cut.tsv')
        if filename == 'test':
            files = [file_3]
            input_file = os.path.join('data', 'eval_512_cut.tsv')

        self.preprocess(input_file, vocab_file, files)
        #self.batch_data = self.read_process(files, batch_size, num_threads, num_gpus, num_epochs)

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
                content = line[0].split()
                if not content:
                    continue
                num += 1
                label = line[1]
                sequence = []
                if len(content) == 512:
                    sequence = [vocab[char] if char in vocab else vocab['unk'] for char in content]
                sequence = np.array(sequence)
                if label == 'POSITIVE':
                    label_sequence = np.array([0, 1])
                else:
                    label_sequence = np.array([1, 0])
                ex = make_example(sequence, label_sequence)
                writer_1.write(ex.SerializeToString())
                if num % 1000 == 0:
                    print (num)
        writer_1.close()
        print (num, self.vocab_size)
        print("Wrote to {}".format(write_to_file_1))

    def read_process(self, filename, batch_size, num_threads, num_gpus, max_epochs):
        reader = tf.TFRecordReader()
        file_queue = tf.train.string_input_producer(filename, shuffle=False, num_epochs=max_epochs)
        key, serialized_example = reader.read(file_queue)
        sequence_features = {
            "inputs": tf.FixedLenSequenceFeature([], dtype=tf.int64),
            "labels": tf.FixedLenSequenceFeature([], dtype=tf.int64)
        }
        # Parse the example (returns a dictionary of tensors)
        _, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features=sequence_features
        )
        input_tensors = [sequence_parsed['inputs'], sequence_parsed['labels']]
        return tf.train.batch(
            input_tensors,
            batch_size=batch_size,
            capacity=10 + num_gpus * batch_size,
            num_threads=num_threads,
            dynamic_pad=True,
            allow_smaller_final_batch=False
        )

if __name__ == "__main__":
    #rand_file()
    write_to_file()
    check()
    #loader = TextLoader('test')
