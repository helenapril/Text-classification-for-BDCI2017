import tensorflow as tf
import os
from data_reader import TextLoader
import codecs
from text_attention import HANClassifierModel


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1.0, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")
tf.flags.DEFINE_string('train_dir', 'check_cnn', 'checkpoint')
tf.flags.DEFINE_string('mode', 'cnn', 'algorithm')
tf.flags.DEFINE_integer('gpu_id', '1', 'algorithm')
tf.flags.DEFINE_string('cell', 'cnn', 'algorithm')
tf.flags.DEFINE_string('up', 0.5, 'algorithm')

FLAGS = tf.flags.FLAGS


def main(_):

    '''model for validation model'''
    ''' build graph for validation and testing (shares parameters with the training graph!) '''
    valid_loader = TextLoader('valid', FLAGS.batch_size, 4, 1, None)
    batch_queue_data = valid_loader.batch_data

    with tf.variable_scope("HANClassifierMode_%s_%s" % (FLAGS.mode, FLAGS.cell)):
        valid_attention = HANClassifierModel(
            batch_queue_data[0],
            batch_queue_data[1],
            num_seq=32,
            num_classes=2,
            vocab_size=50000,
            embedding_size=FLAGS.embedding_dim,
            batch_size=FLAGS.batch_size,
            sentence_size=16,
            cell_name=FLAGS.cell,
            hidden_size=FLAGS.hidden_size,
            dropout_keep_proba=FLAGS.dropout_keep_prob,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            choice=FLAGS.mode,
            up=FLAGS.up,
            dropout_mode=False
        )

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        sess = tf.Session(config=session_conf)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt:
            files = os.listdir(FLAGS.train_dir)
            files = [os.path.join(FLAGS.train_dir, f) for f in files]
            check_point_files = []
            for f in files:
                if '.index' in f:
                    f = f[0:len(f)-6]
                    check_point_files.append(f)
            num_batches_valid = int(5000 / FLAGS.batch_size)
            best_loss = None
            best_model = None
            for check_point in check_point_files:
                saver.restore(sess, check_point)
                global_step = check_point.split('/')[-1].split('-')[-1]
                global_step = int(global_step)
                print ("load model from ", check_point)
                avg_loss = 0
                above = 0
                below = 0
                below_recall = 0
                for step in range(num_batches_valid):
                    loss, ab, ac, ad = sess.run([
                        valid_attention.loss,
                        valid_attention.above,
                        valid_attention.below,
                        valid_attention.below_recall])
                    avg_loss += loss
                    above += ab
                    below += ac
                    below_recall += ad
                acc = float(above) / below
                rec = float(above) / below_recall
                F1 = 2.0 / (1.0 / acc + 1.0 / rec)
                print(global_step, avg_loss/num_batches_valid, acc, rec, F1)
                if best_loss is None or best_loss < F1:
                    best_loss = F1
                    best_model = global_step
            print ('best_model from step %d, epoch:%d, F1: %f' % (best_model, best_model, best_loss))


if __name__ == "__main__":
    tf.app.run()
