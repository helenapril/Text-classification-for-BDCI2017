# -*- coding: utf-8 -*-

import tensorflow as tf


class TRNNConfig(object):
    """RNN配置参数"""

    # 模型参数
    embedding_dim = 64      # 词向量维度
    seq_length = 600        # 序列长度
    num_classes = 2        # 类别数
    vocab_size = 7000       # 词汇表达小
    batch_size = 500

    num_layers = 1           # 隐藏层层数
    hidden_dim = 128        # 隐藏层神经元
    rnn = 'lstm'             # lstm 或 gru


class TextRNN(object):
    """文本分类，RNN模型"""
    def __init__(self, config, input_x, input_y, keep_prob=0.5):
        self.config = config

        # 三个待输入的数据
        self.input_x = input_x
        self.input_y = input_y
        self.keep_prob = keep_prob

        self.rnn()

    def rnn(self):
        """rnn模型"""

        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def gru_cell():  # gru核
            return tf.contrib.rnn.GRUCell(self.config.hidden_dim)

        def dropout(): # 为每一个rnn核后面加一个dropout层
            if (self.config.rnn == 'lstm'):
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
            embedding_inputs = tf.reshape(
                embedding_inputs,
                shape=[self.config.batch_size, self.config.seq_length, self.config.embedding_dim])
            print (embedding_inputs)

        with tf.name_scope("context"):
            with tf.variable_scope("BiLSTM", initializer=tf.orthogonal_initializer()):
                # 多层rnn网络
                cells = [dropout() for _ in range(self.config.num_layers)]
                rnn_cell_fw = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                rnn_cell_bw = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
                outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
                    cell_bw=rnn_cell_bw,
                    cell_fw=rnn_cell_fw,
                    inputs=embedding_inputs,
                    sequence_length=tf.constant(self.config.seq_length, shape=[self.config.batch_size]),
                    dtype=tf.float32)
                outputs = tf.concat(outputs, 2)
                last = outputs[:, -1, :]  # 取最后一个时序输出作为结果

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(last, 64, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("loss"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope("accuracy"):
            # 准确率
            below = tf.reduce_sum(self.y_pred_cls)
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            correct_pred = tf.cast(correct_pred, tf.int64)
            above = tf.reduce_sum(tf.multiply(correct_pred, self.y_pred_cls))
            self.below = below
            self.above = above
            self.accuracy = tf.divide(above, below)

        with tf.name_scope("recall"):
            below = tf.reduce_sum(tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(self.y_pred_cls, tf.argmax(self.input_y, 1))
            correct_predictions = tf.cast(correct_predictions, dtype=tf.int64)
            above = tf.reduce_sum(tf.multiply(correct_predictions, self.y_pred_cls))
            self.below_recall = below
            self.recall = tf.divide(above, below)