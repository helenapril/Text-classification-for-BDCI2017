import tensorflow as tf


def linear(input_, output_size, scope=None):
    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix",
                                 initializer=tf.random_uniform([output_size, input_size], -1.0, 1.0),
                                 dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term


def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope='highway_lin_%d' % idx))
            t = tf.sigmoid(linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1. - t) * input_
            input_ = output

    return input_


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, input_x, input_y, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, dropout_keep_prob, l2_reg_lambda=0.001):

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
        self.input_y = input_y
        self.input_x = input_x

        # Embedding layer [batch_size, sequence_length, embedding_size, 1]
        with tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -0.05, 0.05),
                name="embedding_W")
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_size])
            self.embedding_init = self.W.assign(self.embedding_placeholder)
            self.embedded_word = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_word_expanded = tf.expand_dims(self.embedded_word, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer  [batch_size, len, 1, num_filters]
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="conv_b")
                l2_loss += tf.nn.l2_loss(W)
                conv = tf.nn.conv2d(
                    self.embedded_word_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs [batch_size, 1, 1, num_filters]
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features [batch_size, 1, 1, num_filters_total]
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, dropout_keep_prob)

        #self.highway = highway(self.h_drop, num_filters_total, num_layers=2)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "output_W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="output_b")
            l2_loss += tf.nn.l2_loss(W)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
            self.loss_bias = l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            below = tf.reduce_sum(self.predictions)
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            correct_predictions = tf.cast(correct_predictions, dtype=tf.int64)
            above = tf.reduce_sum(tf.multiply(correct_predictions, self.predictions))
            self.below = below
            self.above = above

        with tf.name_scope("recall"):
            below = tf.reduce_sum(tf.argmax(self.input_y, 1))
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            correct_predictions = tf.cast(correct_predictions, dtype=tf.int64)
            above = tf.reduce_sum(tf.multiply(correct_predictions, self.predictions))
            self.below_recall = below
            self.recall = tf.divide(above, below)

