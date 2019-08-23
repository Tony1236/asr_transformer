# coding: utf-8
# 训练基类，包括多GPU并行运算等
import os
import random
import re
import sys

import tensorflow as tf
from tensorflow.python.ops import init_ops

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))
from nn_utils import learning_rate_decay, average_gradients


class Model(object):
    def __init__(self, hp, num_gpu):
        super(Model, self).__init__()
        self._hp = hp
        self.num_gpu = num_gpu
        self.graph = tf.Graph()

        self._devices = ['/gpu:%d' % i for i in range(num_gpu)] if num_gpu > 0 else ['/cpu:0']
        self.src_pls = tuple()
        self.dst_pls = tuple()

        self.preds, self.istarget = None, None
        self.mean_loss, self.train_op = None, None
        self.test_distance, self.test_length = 0, 0
        self.distance, self.length = 0, 0

        self.global_step, self.learning_rate, self._optimizer = self.prepare_training()
        self._initializer = init_ops.variance_scaling_initializer(scale=1.0, mode='fan_avg', distribution='uniform')

    def prepare_training(self):
        with self.graph.as_default():
            # Optimizer
            global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                                          trainable=False, initializer=tf.zeros_initializer)

            learning_rate = tf.convert_to_tensor(self._hp.train.learning_rate, dtype=tf.float32)
            if self._hp.train.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-8)
            elif self._hp.train.optimizer == 'adam_decay':
                learning_rate *= learning_rate_decay(self._hp, global_step)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
            elif self._hp.train.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif self._hp.train.optimizer == 'mom':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
            else:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

            return global_step, learning_rate, optimizer

    def make_parallel(self, reuse=None, is_training=True, get_wer=False):
        if len(self.src_pls) <= 1:
            raise ValueError

        with self.graph.as_default():
            loss_list, gv_list = [], []
            preds_list, istarget_list = [], []
            cache = {}
            load = dict([(d, 0) for d in self._devices])
            for i, (X, Y, device) in enumerate(zip(self.src_pls, self.dst_pls, self._devices)):

                def daisy_chain_getter(getter, name, *args, **kwargs):
                    """Get a variable and cache in a daisy chain."""
                    device_var_key = (device, name)
                    if device_var_key in cache:
                        # if we have the variable on the correct device, return it.
                        return cache[device_var_key]
                    if name in cache:
                        # if we have it on a different device, copy it from the last device
                        v = tf.identity(cache[name])
                    else:
                        var = getter(name, *args, **kwargs)
                        v = tf.identity(var._ref())  # pylint: disable=protected-access
                    # update the cache
                    cache[name] = v
                    cache[device_var_key] = v
                    return v

                def balanced_device_setter(op):
                    """Balance variables to all devices."""
                    if op.type in {'Variable', 'VariableV2', 'VarHandleOp'}:
                        # return self._sync_device
                        min_load = min(load.values())
                        min_load_devices = [d for d in load if load[d] == min_load]
                        chosen_device = random.choice(min_load_devices)
                        load[chosen_device] += op.outputs[0].get_shape().num_elements()
                        return chosen_device
                    return device

                device_setter = balanced_device_setter
                if i > 0:
                    reuse = True

                with tf.variable_scope(tf.get_variable_scope(),
                                       initializer=self._initializer,
                                       custom_getter=daisy_chain_getter,
                                       reuse=reuse):
                    with tf.device(device_setter):
                        loss, preds, istarget = self.model_loss(X, Y, reuse=reuse, is_training=is_training, scope="transformer")
                        loss_list.append(loss)
                        preds_list.append(preds)
                        istarget_list.append(istarget)
                        var_list = tf.trainable_variables()
                        if self._hp.train.var_filter:
                            var_list = [v for v in var_list if re.match(self._hp.train.var_filter, v.name)]
                        gv_list.append(self._optimizer.compute_gradients(loss, var_list=var_list))
                        if get_wer:
                            distance, length = self.get_wer(Y, preds, index_same=True)
                            self.distance += distance
                            self.length += length

            loss = tf.reduce_mean(loss_list)

            # Clip gradients and then apply.
            grads_and_vars = average_gradients(gv_list)

            if self._hp.train.grads_clip > 0:
                grads, grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                           clip_norm=self._hp.train.grads_clip)
                grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
            else:
                grads_norm = tf.global_norm([gv[0] for gv in grads_and_vars])

            train_op = self._optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
            preds = tf.concat(preds_list, 0)
            istarget = tf.concat(istarget_list, 0)

            return loss, grads_norm, train_op, preds, istarget

    def make_parallel_beam_search(self, reuse=True, is_training=False, get_wer=False):
        if len(self.src_pls) <= 1:
            raise ValueError

        with self.graph.as_default():
            prediction_list = []
            for i, (X, Y, device) in enumerate(zip(self.src_pls, self.dst_pls, self._devices)):
                if i > 0:
                    reuse = True
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse, initializer=self._initializer):
                    with tf.device(device):
                        def true_fn():
                            prediction = self.model_beam_search_preds(X, reuse, is_training, scope="transformer")
                            return prediction

                        def false_fn():
                            return tf.zeros([0, 0], dtype=tf.int32)

                        prediction = tf.cond(tf.greater(tf.shape(X)[0], 0), true_fn, false_fn)
                        if get_wer:
                            test_distance, test_length = self.get_wer(Y, prediction)
                            self.test_distance += test_distance
                            self.test_length += test_length

                        prediction_list.append(prediction)

            max_length = tf.reduce_max([tf.shape(pred)[1] for pred in prediction_list])

            def pad_to_max_length(input, length):
                """Pad the input (with rank 2) with 3(</S>) to the given length in the second axis."""
                shape = tf.shape(input)
                padding = tf.ones([shape[0], length - shape[1]], dtype=tf.int32) * 3
                return tf.concat([input, padding], axis=1)

            prediction_list = tf.concat([pad_to_max_length(pred, max_length) for pred in prediction_list], 0)
            return prediction_list

    def model_output(self, x, y, reuse, is_training, scope):
        raise NotImplementedError()

    def model_loss(self, x, y, reuse, is_training, scope):
        raise NotImplementedError()

    def model_beam_search_preds(self, x, reuse, is_training, scope):
        raise NotImplementedError()

    def get_wer(self, y, preds, index_same=False):
        raise NotImplementedError()