#!/usr/bin/python
#-*-coding=utf-8-*-

import os
import numpy as np
import tensorflow as tf


class TextLoader():
    def __init__(self, filename, batch_size, num_threads, num_gpus, num_epochs):

        file_1 = os.path.join('data', 'train_512_tfrecord')
        file_2 = os.path.join('data', 'valid_512_tfrecord')
        file_3 = os.path.join('data', 'eval_512_tfrecord')
        if filename == 'train':
            files = [file_1]
        if filename == 'valid':
            files = [file_2]
        if filename == 'test':
            files = [file_3]
        self.batch_data = self.read_process(files, batch_size, num_threads, num_gpus, num_epochs)

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
    loader = TextLoader('test', 10, 4, 1, 7)
    batch_queue_data = loader.batch_data

    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for _ in range(2):
        batch_data = sess.run(batch_queue_data)
        input = np.array(batch_data[0])
        print (input.shape)
