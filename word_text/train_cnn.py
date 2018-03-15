import tensorflow as tf
import os
import time
import numpy as np
from data_reader import TextLoader
from text_cnn import TextCNN
from gensim.models.word2vec import Word2Vec

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.000, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("sequence_length", 512, "the number of words")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 80, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 80, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float('max_grad_norm',       5.0,  'normalize gradients at')
tf.flags.DEFINE_float('learning_rate',       0.001,  'starting learning rate')
tf.flags.DEFINE_float('learning_rate_decay', 0.5,  'learning rate decay')
tf.flags.DEFINE_integer('decay_steps', 30000,  'learning rate decay')
tf.flags.DEFINE_string('train_dir', 'check_cnn', 'checkpoint')

FLAGS = tf.flags.FLAGS


def load_word():
    model = Word2Vec.load('data/w2v_model.save')
    weights = model.wv.syn0
    embedding = np.array(weights)
    return embedding


def main(_):

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    train_loader = TextLoader('train', FLAGS.batch_size, 4, 1, None)
    batch_queue_data = train_loader.batch_data
    pretrained_embedding = load_word()
    print("Loading data...")
    # Training
    # ==================================================
    '''define graph'''
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    num_batches_train = int(595000 / (FLAGS.batch_size * FLAGS.num_gpus))
    print ('num_batches_train: %d' % num_batches_train)

    decay_steps = FLAGS.decay_steps
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    with tf.variable_scope('text_cnn'):
        train_cnn = TextCNN(
            batch_queue_data[0],
            batch_queue_data[1],
            sequence_length=FLAGS.sequence_length,
            num_classes=2,
            vocab_size=50000,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            dropout_keep_prob=FLAGS.dropout_keep_prob,
            l2_reg_lambda=FLAGS.l2_reg_lambda)
        gradient, tvar = zip(*opt.compute_gradients(train_cnn.loss + train_cnn.loss_bias))
        gradient, _ = tf.clip_by_global_norm(gradient, FLAGS.max_grad_norm)
        grads = zip(gradient, tvar)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    sess = tf.Session(config=session_conf)

    # Checkpoint directory.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Initialize all variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(train_cnn.embedding_init, feed_dict={
        train_cnn.embedding_placeholder: pretrained_embedding})
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(FLAGS.num_epochs):
        avg_loss = 0.0
        epoch_start_time = time.time()
        above = 0
        below = 0
        below_recall = 0
        for step in range(num_batches_train):
            loss, loss_bias, ab, ac, ad, _, g_step = sess.run([
                train_cnn.loss,
                train_cnn.loss_bias,
                train_cnn.above,
                train_cnn.below,
                train_cnn.below_recall,
                apply_gradient_op,
                global_step])

            avg_loss += loss
            avg_cur_loss = avg_loss / (step + 1)
            above += ab
            below += ac
            below_recall += ad
            if (step + 1) % 200 == 0:
                if ac and ad:
                    accuracy = float(ab) / ac
                    recall = float(ab) / ad
                    if accuracy and recall:
                        F1 = 2.0 / (1.0 / accuracy + 1.0 / recall)
                        print('%d: [%d/%d], batch_train_loss =%.4f, acc = %.4f, rec = %.4f, F1 = %.4f '
                              % (epoch, step + 1, num_batches_train, loss, accuracy, recall, F1))
                    else:
                        print('%d: [%d/%d], batch_train_loss =%.4f, acc = %.4f, rec = %.4f'
                              % (epoch, step + 1, num_batches_train, loss, accuracy, recall))
                else:
                    print (ab, ac, ad)

                print('%d: [%d/%d], train_loss =%.4f'
                      % (epoch, step + 1, num_batches_train, avg_cur_loss))
                print (above, below, below_recall, float(above)/below, float(above)/below_recall)

        print("at the end of epoch:", epoch)
        print('Epoch training time:', time.time() - epoch_start_time)

        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=epoch)
        print('Saved model', checkpoint_path)


if __name__ == "__main__":
    tf.app.run()
