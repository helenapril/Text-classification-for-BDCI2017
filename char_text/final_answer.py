import tensorflow as tf
import os
from data_reader import TextLoader
from text_attention import HANClassifierModel
import codecs
import csv


# Parameters`~``
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


tf.flags.DEFINE_string('output', 'answer1.csv', 'name')
tf.flags.DEFINE_float('up', 0.5, 'checkpoint')
tf.flags.DEFINE_integer('gpu_id', 0, 'name')
tf.flags.DEFINE_string('mode', 'rnn', 'name')

FLAGS = tf.flags.FLAGS


def read1():
    test_loader = TextLoader('test', FLAGS.batch_size, 4, 1, 2)
    batch_queue_data = test_loader.batch_data
    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        test_lstm = HANClassifierModel(
            batch_queue_data[0],
            batch_queue_data[1],
            num_seq=32,
            num_classes=2,
            vocab_size=7000,
            embedding_size=FLAGS.embedding_dim,
            batch_size=FLAGS.batch_size,
            sentence_size=32,
            cell_name='lstm',
            hidden_size=FLAGS.hidden_size,
            dropout_keep_proba=1.0,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            choice=FLAGS.mode,
            dropout_mode=False,
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
        models = [['test_lstm', 'check_attention_lstm_60/model.ckpt-120000'],
                  ['test_lstm', 'check_attention_lstm_60/model.ckpt-138000']]
        answers = []
        for checkpoints in models:
            answer = []
            model = checkpoints[0]
            checkpoint = checkpoints[1]
            global_step = checkpoint.split('/')[-1].split('-')[-1]
            num_batches_test = int(400000 / FLAGS.batch_size)
            saver.restore(sess, checkpoint)
            print ("load model from ", checkpoint)
            if model == 'test_lstm':
                for step in range(num_batches_test):
                    loss, res = sess.run([
                        test_lstm.loss,
                        test_lstm.prediction])
                    answer.extend(res)
                    if (step + 1) % 100 == 0:
                        print (step + 1)
                answers.append(answer)
        sess.close()
    return answers


def read2():
    test_loader = TextLoader('test', FLAGS.batch_size, 4, 1, 2)
    batch_queue_data = test_loader.batch_data
    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        test_highway = HANClassifierModel(
            batch_queue_data[0],
            batch_queue_data[1],
            num_seq=32,
            num_classes=2,
            vocab_size=7000,
            embedding_size=FLAGS.embedding_dim,
            batch_size=FLAGS.batch_size,
            sentence_size=32,
            cell_name='highway',
            hidden_size=FLAGS.hidden_size,
            dropout_keep_proba=1.0,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            choice=FLAGS.mode,
            dropout_mode=False,
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
        models = [['test_highway', 'check_attention_highway_60/model.ckpt-90000'],
                  ['test_highway', 'check_attention_highway_60/model.ckpt-294000']]
        answers = []
        for checkpoints in models:
            answer = []
            model = checkpoints[0]
            checkpoint = checkpoints[1]
            global_step = checkpoint.split('/')[-1].split('-')[-1]
            num_batches_test = int(400000 / FLAGS.batch_size)
            saver.restore(sess, checkpoint)
            print ("load model from ", checkpoint)
            if model == 'test_highway':
                for step in range(num_batches_test):
                    loss, res = sess.run([
                        test_highway.loss,
                        test_highway.prediction])
                    answer.extend(res)
                    if (step + 1) % 100 == 0:
                        print (step + 1)
                answers.append(answer)
    sess.close()
    return answers


def read3():
    test_loader = TextLoader('test', FLAGS.batch_size, 4, 1, 1)
    batch_queue_data = test_loader.batch_data
    with tf.device('/gpu:%d' % FLAGS.gpu_id):
        test_mulint = HANClassifierModel(
            batch_queue_data[0],
            batch_queue_data[1],
            num_seq=32,
            num_classes=2,
            vocab_size=7000,
            embedding_size=FLAGS.embedding_dim,
            batch_size=FLAGS.batch_size,
            sentence_size=32,
            cell_name='MulInt',
            hidden_size=FLAGS.hidden_size,
            dropout_keep_proba=1.0,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            choice=FLAGS.mode,
            dropout_mode=False,
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
        models = [['test_mulint', 'check_attention_mulint_25/model.ckpt-60000']]
        answers = []
        for checkpoints in models:
            answer = []
            model = checkpoints[0]
            checkpoint = checkpoints[1]
            global_step = checkpoint.split('/')[-1].split('-')[-1]
            num_batches_test = int(400000 / FLAGS.batch_size)
            saver.restore(sess, checkpoint)
            print ("load model from ", checkpoint)
            if model == 'test_mulint':
                for step in range(num_batches_test):
                    loss, res = sess.run([
                        test_mulint.loss,
                        test_mulint.prediction])
                    answer.extend(res)
                    if (step + 1) % 100 == 0:
                        print (step + 1)
                answers.append(answer)
    sess.close()
    return answers


def main(_):
    '''model for validation model'''
    ''' build graph for validation and testing (shares parameters with the training graph!) '''
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

    answers = []
    answers.extend(read1())
    #answers.extend(read2())
    #answers.extend(read3())

    with open(FLAGS.output, "wb") as csvfile:
        writer = csv.writer(csvfile)
        for id, res0, res1 in zip(ids, answers[0], answers[1]):
            sum = res0 + res1
            writer.writerow([id, sum])



if __name__ == "__main__":
    tf.app.run()
