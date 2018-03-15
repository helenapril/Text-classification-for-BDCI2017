import tensorflow as tf
import os
from data_reader import TextLoader
from text_cnn import TextCNN
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
tf.flags.DEFINE_integer("sequence_length", 512, "the number of words")
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_gpus", 1, "Batch Size (default: 64)")

tf.flags.DEFINE_string('answer_mode', 'single', 'way to produce answer')
tf.flags.DEFINE_string('load_model', 'model.ckpt-68737', 'name')
tf.flags.DEFINE_string('output', 'answer_cnn.csv', 'name')
tf.flags.DEFINE_string('train_dir', 'check_attention', 'checkpoint')


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

    with tf.variable_scope('text_cnn'):
        test_cnn = TextCNN(
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

    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    sess = tf.Session(config=session_conf)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    saver = tf.train.Saver()

    answer_mode = FLAGS.answer_mode
    if answer_mode == 'single':
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
                    test_cnn.predictions])
                '''for prediction in res:
                    prediction = softmax(prediction)
                    if prediction[1] >= 0.6:
                        answers.append(1)
                    else:
                        answers.append(0)'''
                answers.extend(res)
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

    if answer_mode == 'ensemble':
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        models = [60000, 78000, 48000, 84000, 72000]
        answers = []
        if ckpt:
            for checkpoint in models:
                answer = []
                # saver.restore(sess, ckpt.model_checkpoint_path)
                # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                checkpoints = os.path.join(FLAGS.train_dir, 'model.ckpt-' + str(checkpoint))
                global_step = checkpoints.split('/')[-1].split('-')[-1]
                num_batches_test = int(400000 / FLAGS.batch_size)
                saver.restore(sess, checkpoints)
                print ("load model from ", global_step)
                for step in range(num_batches_test):
                    loss, res = sess.run([
                        test_cnn.loss,
                        test_cnn.predictions])
                    answer.extend(res)
                    if (step + 1) % 100 == 0:
                        print (step + 1)
                answers.append(answer)

            with open(FLAGS.output, "wb") as csvfile:
                writer = csv.writer(csvfile)
                for id, res0, res1, res2, res3, res4 in zip(ids, answers[0], answers[1], answers[2], answers[3], answers[4]):
                    sum = res0 + res1 + res2 + res3 + res4
                    if sum < 3:
                        str_res = 'NEGATIVE'
                    if sum >= 3:
                        str_res = 'POSITIVE'
                    writer.writerow([id, str_res])

if __name__ == "__main__":
     tf.app.run()
