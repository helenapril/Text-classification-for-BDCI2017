import tensorflow as tf
import numpy as np
import os
import time
from data_reader import TextLoader
from text_attention import HANClassifierModel

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
tf.flags.DEFINE_float  ('learning_rate',       0.001,  'starting learning rate')
tf.flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
tf.flags.DEFINE_integer("decay_steps", 3000, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_string('train_dir', 'check_attention', 'checkpoint')
tf.flags.DEFINE_string('mode', 'cnn', 'algorithm')
tf.flags.DEFINE_integer('gpu_id', '0', 'algorithm')
tf.flags.DEFINE_string('cell', 'cnn', 'algorithm')
tf.flags.DEFINE_float ('up', 0.5,  'learning rate decay')
tf.flags.DEFINE_boolean('dropout_mode', False, 'whether to share across time-step')


FLAGS = tf.flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    train_loader = TextLoader('train', FLAGS.batch_size, 4, 1, None)
    batch_queue_data = train_loader.batch_data
    print("Loading data...")
    # Training
    # ==================================================
    '''define graph'''
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    num_batches_train = int(600000 / (FLAGS.batch_size * FLAGS.num_gpus))
    print ('num_batches_train: %d' % num_batches_train)

    decay_steps = FLAGS.decay_steps
    '''lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay,
                                    staircase=True)'''
    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        train_cnn = HANClassifierModel(
            batch_queue_data[0],
            batch_queue_data[1],
            num_seq=32,
            num_classes=2,
            vocab_size=50000,
            embedding_size=FLAGS.embedding_dim,
            batch_size=FLAGS.batch_size,
            sentence_size=32,
            cell_name=FLAGS.cell,
            hidden_size=FLAGS.hidden_size,
            dropout_keep_proba=FLAGS.dropout_keep_prob,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            choice=FLAGS.mode,
            up=FLAGS.up,
            dropout_mode=FLAGS.dropout_mode
        )
        gradient, tvar = zip(*opt.compute_gradients(train_cnn.loss))
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
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(FLAGS.num_epochs):
            loss = 0.0
            epoch_start_time = time.time()
            above = 0
            below = 0
            below_recall = 0
            for step in range(num_batches_train):
                batch_loss, ab, ac, ad, _, g_step = sess.run([
                    train_cnn.loss,
                    train_cnn.above,
                    train_cnn.below,
                    train_cnn.below_recall,
                    apply_gradient_op,
                    global_step])

                loss += batch_loss
                avg_cur_loss = loss / (step + 1)
                above += ab
                below += ac
                below_recall += ad
                if (step + 1) % 100 == 0:
                    if ac and ad:
                        accuracy = float(ab) / ac
                        recall = float(ab) / ad
                        if accuracy and recall:
                            F1 = 2.0 / (1.0 / accuracy + 1.0 / recall)
                            print('%d: [%d/%d], batch_train_loss =%.4f, acc = %.4f, rec = %.4f, F1 = %.4f '
                                % (epoch, step + 1, num_batches_train, batch_loss, accuracy, recall, F1))
                        else:
                            print('%d: [%d/%d], batch_train_loss =%.4f, acc = %.4f, rec = %.4f'
                                  % (epoch, step + 1, num_batches_train, batch_loss, accuracy, recall))
                    else:
                        print('%d: [%d/%d], batch_train_loss =%.4f, ab = %d, ac = %d, ad = %d '
                              % (epoch, step + 1, num_batches_train, batch_loss, ab, ac, ad))
                    if below and below_recall:
                        accuracy = float(above) / below
                        recall = float(above) / below_recall
                        if accuracy and recall:
                            f1 = 2.0/(1.0/accuracy + 1.0/recall)
                            print (above, below, below_recall, accuracy, recall, f1)
                        else:
                            print (above, below, below_recall, accuracy, recall)
                    else:
                        print (above, below, below_recall)
                    print (avg_cur_loss)

            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)

            print("at the end of epoch:", epoch)
            print('Epoch training time:', time.time() - epoch_start_time)


if __name__ == "__main__":
    tf.app.run()