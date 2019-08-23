# coding: utf-8
import difflib
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.session_bundle.exporter as exporter


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def expand_feed_dict(feed_dict):
    """If the key is a tuple of placeholders,
    split the input data then feed them into these placeholders.
    """
    new_feed_dict = {}
    for k, v in feed_dict.items():
        if type(k) is not tuple:
            new_feed_dict[k] = v
        else:
            # Split v along the first dimension.
            n = len(k)
            batch_size = v.shape[0]
            span = batch_size // n
            remainder = batch_size % n
            # assert span > 0
            base = 0
            for i, p in enumerate(k):
                if i < remainder:
                    end = base + span + 1
                else:
                    end = base + span
                new_feed_dict[p] = v[base: end]
                base = end
    return new_feed_dict


def GetEditDistance(str1, str2):
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        # print('{:7} a[{}: {}] --> b[{}: {}] {} --> {}'.format(tag, i1, i2, j1, j2, str1[i1: i2], str2[j1: j2]))
        if tag == 'replace':
            leven_cost += max(i2 - i1, j2 - j1)
        elif tag == 'insert':
            leven_cost += (j2 - j1)
        elif tag == 'delete':
            leven_cost += (i2 - i1)
    return leven_cost


def batch_edit_distance(logits, preds, istargets):
    distance = 0
    length = 0
    for index, logit in enumerate(logits):
        istarget_num = int(sum(istargets[index]))
        pred = preds[index][:istarget_num]
        logit = logit[:istarget_num]
        d = GetEditDistance(pred, logit)
        if d > istarget_num:
            distance += istarget_num
        else:
            distance += d
        length += istarget_num
    return distance, length


def model_save(sess, path, model_name, global_step):
    """
    模型保存
    :param sess: tf.Session()
    :param path: 模型保存的路径
    :param model_name: 模型保存的名称
    :param global_step: 模型保存的迭代数
    :return:
    """
    # 模型保存
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(path, model_name), global_step=global_step)
    return saver


def dense(inputs,
          output_size,
          activation=tf.identity,
          use_bias=True,
          reuse_kernel=None,
          reuse=None,
          name=None):
    argcount = activation.func_code.co_argcount
    if activation.func_defaults:
        argcount -= len(activation.func_defaults)
    assert argcount in (1, 2)
    with tf.variable_scope(name, "dense", reuse=reuse):
        if argcount == 1:
            input_size = inputs.get_shape().as_list()[-1]
            inputs_shape = tf.unstack(tf.shape(inputs))
            inputs = tf.reshape(inputs, [-1, input_size])
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_kernel):
                w = tf.get_variable("kernel", [output_size, input_size])
            outputs = tf.matmul(inputs, w, transpose_b=True)
            if use_bias:
                b = tf.get_variable("bias", [output_size], initializer=tf.zeros_initializer)
                outputs += b
            outputs = activation(outputs)
            return tf.reshape(outputs, inputs_shape[:-1] + [output_size])
        else:
            arg1 = dense(inputs, output_size, tf.identity, use_bias, name='arg1')
            arg2 = dense(inputs, output_size, tf.identity, use_bias, name='arg2')
            return activation(arg1, arg2)


# 转化一个序列列表为稀疏矩阵
def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), xrange(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def edit_distance_loss(y, preds):
    zero = tf.constant(0, dtype=tf.int32)
    where = tf.not_equal(y, zero)
    indices = tf.where(where)
    y_sparse = dense_to_sparse(y, indices)
    preds_sparse = dense_to_sparse(preds, indices)
    loss = tf.edit_distance(preds_sparse, y_sparse, True)
    return tf.reduce_mean(loss)


def dense_to_sparse(dense_value, indices):
    values = tf.gather_nd(dense_value, indices)
    sparse = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(dense_value)))
    return sparse


def edit_distance_wer(y, preds, index_same=False):
    where_y = tf.logical_and(tf.not_equal(y, tf.constant(0, dtype=tf.int32)),
                             tf.not_equal(y, tf.constant(3, dtype=tf.int32)))
    indices_y = tf.where(where_y)
    if index_same:
        indices_preds = indices_y
    else:
        indices_preds = tf.where(tf.logical_and(tf.not_equal(preds, tf.constant(0, dtype=tf.int32)),
                                                tf.not_equal(preds, tf.constant(3, dtype=tf.int32))))
    y_sparse = dense_to_sparse(y, indices_y)
    preds_sparse = dense_to_sparse(preds, indices_preds)
    distance = tf.reduce_sum(tf.edit_distance(preds_sparse, y_sparse, False))
    length = tf.reduce_sum(tf.to_int32(where_y))
    return distance, length


def exporter_model(saver, sess, work_dir, export_version, x, y):
    """
    :param saver: tf.train.Saver()
    :param sess: tf.Session()
    :param work_dir: 保存的路径
    :param export_version: 保存的版本数，tensorflow serving会优先读取最高的版本
    :param x: 模型 input
    :param y: 模型 predict result
    :return:
    """
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': x}),
            'outputs': exporter.generic_signature({'y': y})})
    model_exporter.export(work_dir,
                          tf.constant(export_version), sess)


if __name__ == '__main__':
    y = [[1, 2, 3, 0, 0], [4, 5, 3, 0, 0], [4, 4, 5, 6, 3]]
    preds = [[1, 2, 3, 3, 3], [4, 5, 3, 3, 6], [4, 4, 5, 3, 3]]

    y = tf.Variable(np.array(y), dtype=tf.int32)
    preds = tf.Variable(np.array(preds), dtype=tf.int32)
    d, l = edit_distance_wer(y, preds, index_same=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print sess.run(d)
