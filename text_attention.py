import tensorflow as tf
import tensorflow.contrib.layers as layers

try:
    from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple


def bidirectional_rnn(cell_fw, cell_bw, inputs_embedded, input_lengths,
                      scope=None):
    """Bidirecional RNN with concatenated outputs and states
        inputs_embedded : [batch_size , sentence_legth, word_embed_size]]
        return : [batch_size , sentence_length, 2*hidden_layer_size]"""
    with tf.variable_scope(scope or "birnn", initializer=tf.orthogonal_initializer()) as scope:
        outputs, output_states = (
            tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                            cell_bw=cell_bw,
                                            inputs=inputs_embedded,
                                            sequence_length=input_lengths,
                                            dtype=tf.float32,
                                            scope=scope))
        outputs = tf.concat(outputs, 2)
        return outputs


def task_specific_attention(inputs, output_size,
                            initializer=layers.xavier_initializer(),
                            activation_fn=tf.tanh, scope=None):
    """
    Performs task-specific attention reduction, using learned
    attention context vector (constant within task of interest).
    Args:
        inputs: [batch_size , sentence_length, 2*hidden_layer_size]
        output_size: Size of output's inner (feature) dimension
    Returns:
        outputs: Tensor of shape [batch_size, output_dim].
    """
    assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

    with tf.variable_scope(scope or 'attention') as scope:
        attention_context_vector = tf.get_variable(name='attention_context_vector',
                                                   shape=[output_size],
                                                   initializer=initializer,
                                                   dtype=tf.float32)
        input_projection = layers.fully_connected(inputs, output_size,
                                                  activation_fn=activation_fn,
                                                  scope=scope)

        vector_attn = tf.reduce_sum(tf.multiply(input_projection, attention_context_vector), axis=2, keep_dims=True)
        attention_weights = tf.nn.softmax(vector_attn, dim=1)
        weighted_projection = tf.multiply(inputs, attention_weights)

        outputs = tf.reduce_sum(weighted_projection, axis=1)

        return outputs


class HANClassifierModel():

    def __init__(self, inputs, labels, vocab_size, embedding_size, num_classes, num_seq, batch_size,
                 sentence_size, cell_name, hidden_size,
                 dropout_keep_proba, filter_sizes, num_filters, choice):

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_seq = num_seq
        self.batch_size = batch_size
        self.sentence_size = sentence_size
        self.dropout_keep_proba = dropout_keep_proba
        self.cell_name = cell_name
        self.hidden_size = hidden_size

        # [document x sentence x word]
        self.inputs = tf.reshape(inputs, shape=[batch_size, num_seq*sentence_size])

        # [document]
        self.labels = labels

        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.get_variable(name="embedding_matrix",
                                                    shape=[self.vocab_size, self.embedding_size],
                                                    initializer=tf.random_normal_initializer(-0.05, 0.05),
                                                    dtype=tf.float32)
            self.inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True, forget_bias=1.0)

        def gru_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_size)

        def dropout():
            if self.cell_name == 'lstm':
                cell = lstm_cell()
            else:
                cell = gru_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_proba)

        with tf.variable_scope('hierarchical_attention'):
            if choice == 'cnn':
                self.inputs_embedded_expanded = tf.expand_dims(self.inputs_embedded, -1)
                self.inputs_embedded_expanded = tf.reshape(self.inputs_embedded_expanded,
                                                           [self.batch_size * self.num_seq, self.sentence_size,
                                                            self.embedding_size, -1])
                pooled_outputs = []
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, embedding_size, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
                        conv = tf.nn.conv2d(
                            self.inputs_embedded_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs [batch_size, 1, 1, num_filters]
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, sentence_size - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features [batch_size, 1, 1, num_filters_total]
                self.num_filters_total = num_filters * len(filter_sizes)
                self.h_pool = tf.concat(pooled_outputs, 3)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])
                self.word_level_output = tf.nn.dropout(self.h_pool_flat, dropout_keep_proba)

            if choice == 'rnn':
                cells = dropout()
                word_level_inputs = tf.reshape(self.inputs_embedded,
                                               [self.batch_size * self.num_seq, self.sentence_size,
                                                self.embedding_size])
                self.word_lengths = tf.constant(32, shape=[batch_size * num_seq])
                word_encoder_output = bidirectional_rnn(cells, cells,
                                                        word_level_inputs, self.word_lengths)
                self.word_level_output = task_specific_attention(
                    word_encoder_output, 100)

            with tf.variable_scope('sent_attention') as scope:
                cells = dropout()
                if choice == 'cnn':
                    sentence_inputs = tf.reshape(self.word_level_output,
                                             [self.batch_size, self.num_seq, self.num_filters_total])
                else:
                    sentence_inputs = tf.reshape(self.word_level_output,
                                                 [self.batch_size, self.num_seq, 2 * self.hidden_size])
                sentence_encoder_output = bidirectional_rnn(
                    cells, cells, sentence_inputs,
                    tf.constant(self.num_seq, shape=[self.batch_size]), scope=scope)

                sentence_level_output = task_specific_attention(
                    sentence_encoder_output, 100, scope=scope)
                self.sentence_level_output = tf.nn.dropout(sentence_level_output, dropout_keep_proba)

                self.logits = layers.fully_connected(
                        sentence_level_output, self.num_classes, activation_fn=None)

                self.prediction = tf.argmax(self.logits, axis=-1)

            with tf.name_scope("loss"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=self.labels)
                self.loss = tf.reduce_mean(cross_entropy)

            with tf.name_scope("accuracy"):
                below = tf.reduce_sum(self.prediction)
                correct_pred = tf.equal(tf.argmax(self.labels, 1), self.prediction)
                correct_pred = tf.cast(correct_pred, tf.int64)
                above = tf.reduce_sum(tf.multiply(correct_pred, self.prediction))
                self.below = below
                self.above = above
                self.accuracy = tf.divide(above, below)

            with tf.name_scope("recall"):
                below = tf.reduce_sum(tf.argmax(self.labels, 1))
                correct_predictions = tf.equal(self.prediction, tf.argmax(self.labels, 1))
                correct_predictions = tf.cast(correct_predictions, dtype=tf.int64)
                above = tf.reduce_sum(tf.multiply(correct_predictions, self.prediction))
                self.below_recall = below
                self.recall = tf.divide(above, below)




