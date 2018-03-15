import tensorflow as tf
import numpy as np
import os
import time
from data_reader import TextLoader
from text_rnn import TextRNN, TRNNConfig

# Parameters
# ==================================================

tf.flags.DEFINE_integer("batch_size", 500, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 7, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("num_checkpoints", 50, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float  ('max_grad_norm',       5.0,  'normalize gradients at')
tf.flags.DEFINE_float  ('learning_rate',       0.1,  'starting learning rate')
tf.flags.DEFINE_float  ('learning_rate_decay', 0.5,  'learning rate decay')
tf.flags.DEFINE_integer ('decay_steps', 300,  'learning rate decay')
tf.flags.DEFINE_string('train_dir', 'check_rnn', 'checkpoint')
tf.flags.DEFINE_string('summary_dir', 'summary', 'checkpoint')

FLAGS = tf.flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
        print('Created training directory', FLAGS.train_dir)

    train_loader = TextLoader('train', FLAGS.batch_size, 4, 1, FLAGS.num_epochs)
    batch_queue_data = train_loader.batch_data

    print('Configuring RNN model...')
    config = TRNNConfig()
    train_rnn = TextRNN(config, batch_queue_data[0], batch_queue_data[1])

    # Training
    # ==================================================
    '''define graph'''
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)
    num_batches_train = int(450000 / (FLAGS.batch_size * FLAGS.num_gpus))
    print ('num_batches_train: %d' % num_batches_train)

    decay_steps = FLAGS.decay_steps
    lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay,
                                    staircase=True)
    opt = tf.train.AdamOptimizer(lr)

    gradient, tvar = zip(*opt.compute_gradients(train_rnn.loss))
    gradient, _ = tf.clip_by_global_norm(gradient, FLAGS.max_grad_norm)
    grads = zip(gradient, tvar)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Summaries for loss and accuracy
    #loss_summary = tf.summary.scalar("loss", train_cnn.loss)
    #acc_summary = tf.summary.scalar("accuracy", train_cnn.accuracy)

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    sess = tf.Session(config=session_conf)

    # Train Summaries
    '''train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(FLAGS.summary_dir, "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)'''

    # Checkpoint directory.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

    # Initialize all variables
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for epoch in range(FLAGS.num_epochs):
        avg_loss = 0.0
        epoch_start_time = time.time()
        above = 0
        below = 0
        below_recall = 0
        for step in range(num_batches_train):
            loss, ab, ac, ad, _, = sess.run([
                train_rnn.loss,
                train_rnn.above,
                train_rnn.below,
                train_rnn.below_recall,
                apply_gradient_op])
            #train_summary_writer.add_summary(summaries, step)
            avg_loss += loss
            avg_cur_loss = avg_loss / (step + 1)
            above += ab
            below += ac
            below_recall += ad
            if (step + 1) % 10 == 0:
                print('%d: [%d/%d], train_loss =%.4f'
                      % (epoch, step + 1, num_batches_train, avg_cur_loss))
                print (above, below, below_recall, float(above)/below, float(above)/below_recall)
            if (step + 1) % 500 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
                print('Saved model', checkpoint_path)
        print("at the end of epoch:", epoch)
        print('Epoch training time:', time.time() - epoch_start_time)

        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=global_step)
        print('Saved model', checkpoint_path)


if __name__ == "__main__":
    tf.app.run()