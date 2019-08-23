# coding: utf-8
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import os
import sys
from hyperparams import Hyperparams as hp
hp.num_epochs = 30
from transformer import ExtraTransformer

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))
from read_data_end2end import DataSpeech
from nn_utils import expand_feed_dict, batch_edit_distance, model_save

data_train = DataSpeech('train')
yield_train_data = data_train.yield_singal_batch_data(hp.batch_size, hp.audio_signal_length, hp.maxlen, speed=True)

data_test = DataSpeech('test')
yield_test_data = data_test.yield_singal_batch_data(hp.batch_size, hp.audio_signal_length, hp.maxlen, speed=False)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5"

tf_config = tf.ConfigProto()
# 指定内存占用
tf_config.gpu_options.allow_growth = True
tf_config.allow_soft_placement = True


def train_one_step(feat_batch, target_batch, model, sess):
    feed_dict = expand_feed_dict({model.signal_pls: feat_batch,
                                  model.dst_pls: target_batch})
    step, lr, mean_loss, _, preds, istarget, distance, length = sess.run(
        [model.global_step, model.learning_rate,
         model.mean_loss, model.train_op, model.preds, model.istarget, model.distance, model.length],
        feed_dict=feed_dict)
    return step, lr, mean_loss, preds, istarget, distance, length


def eval(feat_batch, target_batch, model, sess):
    feed_dict = expand_feed_dict({model.signal_pls: feat_batch,
                                  model.dst_pls: target_batch})
    preds, test_distance, test_length = sess.run([model.test_preds, model.test_distance, model.test_length],
                                                 feed_dict=feed_dict)
    return preds, test_distance, test_length


if __name__ == '__main__':
    tmodel = ExtraTransformer(hp, 6, data_train)
    tmodel.build_model(get_wer=True)
    tmodel.build_test_model(get_wer=True)

    num_batch = data_train.get_data_num() // hp.batch_size

    with tf.Session(config=tf_config, graph=tmodel.graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, hp.num_epochs + 1):
            words_num = 0
            word_error_num = 0
            train_loss = 0
            for step in tqdm(range(num_batch * 5), total=num_batch * 5, ncols=70, leave=False,
                             unit='b'):
                x, y, _ = yield_train_data.next()
                step, lr, mean_loss, preds, istarget, distance, length = train_one_step(x, y, tmodel, sess)
                if distance > length: distance = length

                train_loss += mean_loss

                word_error_num += distance
                words_num += length

            mean_loss = train_loss * 1.0 / num_batch / 10
            wer = word_error_num * 1.0 / words_num * 100
            print("epoch: %s, loss: %s, wer: %s" % (epoch, mean_loss, wer))

            words_num = 0
            word_error_num = 0
            for step in tqdm(range(data_test.get_data_num() // hp.batch_size),
                             total=data_test.get_data_num() // hp.batch_size, ncols=70, leave=False, unit='b'):
                x, y, _ = yield_test_data.next()

                preds, test_distance, test_length = eval(x, y, tmodel, sess)
                if test_distance > test_length: test_distance = test_length

                word_error_num += test_distance
                words_num += test_length

            wer = word_error_num * 1.0 / words_num * 100
            print("test_wer: %s" % wer)

            if epoch % hp.save_epochs == 0:
                model_save(sess, current_relative_path(hp.logdir + '/'), "transformer", epoch)