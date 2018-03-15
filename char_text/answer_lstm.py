import tensorflow as tf
import os
from data_reader import TextLoader
from text_attention import HANClassifierModel
import codecs
import csv
import numpy as np


# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (default: 128)")

# Training parameters
tf.flags.DEFINE_integer("sequence_length", 1024, "the number of words")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")

tf.flags.DEFINE_string('mode', 'cnn', 'algorithm')
tf.flags.DEFINE_integer('gpu_id', '1', 'algorithm')
tf.flags.DEFINE_string('load_model', 'model.ckpt-68737', 'name')
tf.flags.DEFINE_string('output', 'answer1.csv', 'name')
tf.flags.DEFINE_string('train_dir', 'check_attention', 'checkpoint')
tf.flags.DEFINE_string('cell', 'cnn', 'algorithm')
tf.flags.DEFINE_float('up', '0.6', 'name')

FLAGS = tf.flags.FLAGS


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def main(_):

    '''model for validation model'''
    ''' build graph for validation and testing (shares parameters with the training graph!) '''
    test_loader = TextLoader('test', FLAGS.batch_size, 4, 1, 1)
    batch_queue_data = test_loader.batch_data
    ids = []
    num = 0
    input_file = os.path.join('data', 'eval.tsv')
    with codecs.open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split('\t')
            if len(line) != 3:
                continue
            id = line[0]
            ids.append(id)
            num += 1
            if num % 10000 == 0:
                print (num)

    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        test_cnn = HANClassifierModel(
            batch_queue_data[0],
            batch_queue_data[1],
            num_seq=32,
            num_classes=2,
            vocab_size=7000,
            embedding_size=FLAGS.embedding_dim,
            batch_size=FLAGS.batch_size,
            sentence_size=32,
            cell_name=FLAGS.cell,
            hidden_size=FLAGS.hidden_size,
            dropout_keep_proba=1.0,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            choice=FLAGS.mode,
            up=FLAGS.up
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
        answers = []
        if ckpt:
            # saver.restore(sess, ckpt.model_checkpoint_path)
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            checkpoints = os.path.join(FLAGS.train_dir, FLAGS.load_model)
            global_step = checkpoints.split('/')[-1].split('-')[-1]
            num_batches_test = int(400000 / FLAGS.batch_size)
            saver.restore(sess, checkpoints)
            print ("load model from ", global_step)
            for step in range(num_batches_test):
                loss, res = sess.run([
                    test_cnn.loss,
                    test_cnn.logits])
                for prediction in res:
                    prediction = softmax(prediction)
                    if prediction[1] >= 0.55:
                        answers.append(1)
                    else:
                        answers.append(0)
                if (step + 1) % 100 == 0:
                    print (step + 1)

        with open(FLAGS.output, "wb") as csvfile:
            writer = csv.writer(csvfile)
            for id, res in zip(ids, answers):
                if res == 0:
                    str_res = 'NEGATIVE'
                if res == 1:
                    str_res = 'POSITIVE'
                writer.writerow([id, str_res])

if __name__ == "__main__":
    tf.app.run()