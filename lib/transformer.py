# -*- coding: utf-8 -*-
import os
import random
import re
import sys

from modules import *

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))

from model import Model
from nn_utils import edit_distance_loss, edit_distance_wer, average_gradients


class Transformer(Model):
    def __init__(self, hp, num_gpu, data):
        super(Transformer, self).__init__(hp, num_gpu)
        self._data = data
        self.placeholders()

    def placeholders(self):
        with self.graph.as_default():
            src_pls = []
            dst_pls = []
            for i, device in enumerate(self._devices):
                with tf.device(device):
                    pls_batch_x = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self._hp.audio_length, self._hp.audio_feature_length],
                                                 name='src_pl_{}'.format(i))  # [batch, feat, feat_dim]
                    pls_batch_y = tf.placeholder(dtype=tf.int32, shape=[None, self._hp.maxlen],
                                                 name='dst_pl_{}'.format(i))  # [batch, len]
                    src_pls.append(pls_batch_x)
                    dst_pls.append(pls_batch_y)
            self.src_pls = tuple(src_pls)
            self.dst_pls = tuple(dst_pls)

    def _transformer_encoder(self, enc, reuse, is_training):
        # Encoder
        with tf.variable_scope("encoder"):

            # Positional Encoding
            if self._hp.sinusoid:
                enc += positional_encoding(enc,
                                           num_units=self._hp.hidden_units,
                                           zero_pad=False,
                                           scale=False,
                                           reuse=reuse,
                                           scope="enc_pe")
            else:
                enc += embedding(
                    tf.tile(tf.expand_dims(tf.range(tf.shape(enc)[1]), 0), [tf.shape(enc)[0], 1]),
                    vocab_size=self._hp.audio_length,
                    num_units=self._hp.hidden_units,
                    zero_pad=False,
                    reuse=reuse,
                    scale=False,
                    scope="enc_pe")

            # Dropout
            enc = tf.layers.dropout(enc,
                                    rate=self._hp.dropout_rate,
                                    training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(self._hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              num_units=self._hp.hidden_units,
                                              num_heads=self._hp.num_heads,
                                              dropout_rate=self._hp.dropout_rate,
                                              is_training=is_training,
                                              reuse=reuse,
                                              causality=False)

                    # Feed Forward
                    enc = feedforward(enc, num_units=[4 * self._hp.hidden_units, self._hp.hidden_units],
                                      reuse=reuse)
        return enc

    def _transformer_decoder(self, enc, decoder_inputs, reuse, is_training):
        # Decoder
        with tf.variable_scope("decoder"):
            # Embedding
            dec = embedding(decoder_inputs,
                            vocab_size=self._data.symbol_num,
                            num_units=self._hp.hidden_units,
                            scale=True,
                            reuse=reuse,
                            scope="dec_embed")

            # Positional Encoding
            if self._hp.sinusoid:
                dec += positional_encoding(decoder_inputs,
                                           num_units=self._hp.hidden_units,
                                           zero_pad=False,
                                           reuse=reuse,
                                           scale=False,
                                           scope="dec_pe")
            else:
                dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(decoder_inputs)[1]), 0),
                                         [tf.shape(decoder_inputs)[0], 1]),
                                 vocab_size=self._hp.maxlen,
                                 num_units=self._hp.hidden_units,
                                 zero_pad=False,
                                 scale=False,
                                 reuse=reuse,
                                 scope="dec_pe")

            # Dropout
            dec = tf.layers.dropout(dec,
                                    rate=self._hp.dropout_rate,
                                    training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(self._hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention ( self-attention)
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              num_units=self._hp.hidden_units,
                                              num_heads=self._hp.num_heads,
                                              dropout_rate=self._hp.dropout_rate,
                                              is_training=is_training,
                                              causality=True,
                                              reuse=reuse,
                                              scope="self_attention")

                    # Multihead Attention ( vanilla attention)
                    dec = multihead_attention(queries=dec,
                                              keys=enc,
                                              num_units=self._hp.hidden_units,
                                              num_heads=self._hp.num_heads,
                                              dropout_rate=self._hp.dropout_rate,
                                              is_training=is_training,
                                              causality=False,
                                              reuse=reuse,
                                              scope="vanilla_attention")

                    # Feed Forward
                    dec = feedforward(dec, num_units=[4 * self._hp.hidden_units, self._hp.hidden_units],
                                      reuse=reuse)
        return dec

    def _transformer_model(self, enc, decoder_inputs, reuse, is_training):
        # Encoder
        enc = self._transformer_encoder(enc, reuse, is_training)
        return self._transformer_decoder(enc, decoder_inputs, reuse, is_training)

    def model_output(self, x, y, reuse, is_training, scope, decoder_inputs=None):
        with tf.variable_scope(scope, reuse=reuse):
            # define decoder inputs
            if decoder_inputs is None:
                decoder_inputs = tf.concat((tf.ones_like(y[:, :1]) * 2, y[:, :-1]), -1)  # 2:<S>
            # feature维度转transformer维度
            enc = tf.layers.dense(x, self._hp.hidden_units, reuse=reuse)
            enc = normalize(enc, reuse=reuse)
            dec = self._transformer_model(enc, decoder_inputs, reuse, is_training)

            with tf.variable_scope("logits", reuse=reuse):
                # Final linear projection
                logits = tf.layers.dense(dec, self._data.symbol_num, reuse=reuse)
        return logits

    def model_loss(self, x, y, reuse, is_training, scope):
        logits = self.model_output(x, y, reuse, is_training, scope)
        y_smoothed = label_smoothing(tf.one_hot(y, depth=self._data.symbol_num))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)
        istarget = tf.to_float(tf.not_equal(y, 0))
        mean_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))

        preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
        if 'mwer' in self._hp.loss_type:
            mwer_loss = self.mwer_loss(y, preds)
            mean_loss = self._hp.train.loss_ce * mean_loss + self._hp.train.loss_mwer * mwer_loss

        if 'l2' in self._hp.loss_type:
            apply_regularization_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(1e-5),
                tf.trainable_variables())
            mean_loss += apply_regularization_loss

        return mean_loss, preds, istarget

    def mwer_loss(self, y, preds):
        mwer_loss = edit_distance_loss(y, preds)
        return mwer_loss

    def model_scheduled_sampling_output(self, x, y, reuse, is_training, scope):
        """为了弥补teacher force训练与推理的不一致性，采用真实推理的内容作为下一次推理的内容"""
        # 采用while_loop训练
        with tf.variable_scope(scope, reuse=reuse):
            enc = tf.layers.dense(x, self._hp.hidden_units, reuse=reuse)
            enc = normalize(enc, reuse=reuse)
            enc = self._transformer_encoder(enc, reuse, is_training)
            preds, logits = self.scheduled_sampling_preds(enc, y, reuse=reuse, is_training=is_training)

        y_smoothed = label_smoothing(tf.one_hot(y, depth=self._data.symbol_num))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_smoothed)
        istarget = tf.to_float(tf.not_equal(y, 0))
        mean_loss = tf.reduce_sum(loss * istarget) / (tf.reduce_sum(istarget))

        if 'mwer' in self._hp.loss_type:
            mwer_loss = self.mwer_loss(y, preds)
            mean_loss = self._hp.train.loss_ce * mean_loss + self._hp.train.loss_mwer * mwer_loss

        if 'l2' in self._hp.loss_type:
            apply_regularization_loss = tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(1e-5),
                tf.trainable_variables())
            mean_loss += apply_regularization_loss

        return mean_loss, preds, istarget

    def get_wer(self, y, preds, index_same=False):
        distance, length = edit_distance_wer(y, preds, index_same=index_same)
        return distance, length

    def get_train_op(self, is_training, reuse=None, get_wer=False):
        if self.num_gpu >= 2:
            loss, grads_norm, train_op, preds, istarget = self.make_parallel(reuse=reuse, is_training=is_training,
                                                                             get_wer=get_wer)
            if not is_training:
                train_op = tf.no_op()
            return loss, train_op, preds, istarget

        with self.graph.as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse, initializer=self._initializer):
                mean_loss, preds, istarget = self.model_loss(self.src_pls[0], self.dst_pls[0], reuse, is_training,
                                                             "transformer")
                if is_training:
                    train_op = self._optimizer.minimize(mean_loss, global_step=self.global_step)
                else:
                    train_op = tf.no_op()

            if get_wer:
                self.distance, self.length = self.get_wer(self.dst_pls[0], preds, index_same=True)

        return mean_loss, train_op, preds, istarget

    def build_model(self, is_training=True, get_wer=False):
        self.mean_loss, self.train_op, self.preds, self.istarget = self.get_train_op(is_training, get_wer=get_wer)

    def get_beam_search_preds(self, is_training=False, reuse=True, get_wer=False):
        if self.num_gpu >= 2:
            return self.make_parallel_beam_search(reuse=reuse, is_training=is_training, get_wer=get_wer)

        with self.graph.as_default():
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse, initializer=self._initializer):
                test_preds = self.model_beam_search_preds(self.src_pls[0], reuse, is_training, "transformer")

            if get_wer:
                self.test_distance, self.test_length = self.get_wer(self.dst_pls[0], test_preds)

        return test_preds

    def model_beam_search_preds(self, x, reuse, is_training, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # feature维度转transformer维度
            enc = tf.layers.dense(x, self._hp.hidden_units, reuse=reuse)
            enc = normalize(enc, reuse=reuse)
            enc_output = self._transformer_encoder(enc, reuse, is_training)
            test_preds = self.beam_search(enc_output, reuse)
        return test_preds

    def build_test_model(self, is_training=False, reuse=True, get_wer=False):
        self.test_preds = self.get_beam_search_preds(reuse=reuse, is_training=is_training, get_wer=get_wer)

    def scheduled_sampling_preds(self, encoder_output, y, reuse, is_training):
        # Prepare beam search inputs.
        batch_size = tf.shape(encoder_output)[0]
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size, 1], dtype=tf.int32) * 2
        # logits
        logits = tf.zeros([batch_size, 0, self._data.symbol_num])

        y_max_length = tf.to_int32(tf.reduce_max(tf.reduce_sum(tf.to_float(tf.not_equal(y, 0)), axis=1)))

        def not_finished(i, preds, logits):
            return tf.less(i, y_max_length)

        def step(i, preds, logits):
            i += 1
            decoder_output = self._transformer_decoder(encoder_output, preds, is_training=is_training, reuse=reuse)

            with tf.variable_scope("logits", reuse=reuse):
                # Final linear projection
                last_logits = tf.layers.dense(decoder_output[:, -1], self._data.symbol_num, reuse=reuse)

            # last_preds = tf.to_int32(tf.arg_max(last_logits, dimension=-1))[:, None]
            # 采样概率正确值或者推理值
            sampler = tf.constant(np.array([[0.5, 0.5]]), dtype=tf.float32)
            sampler_result = tf.multinomial(sampler, 1, seed=None, name=None)

            def teacher_force():
                return y[:, i - 1:i]

            def inference():
                z = tf.nn.log_softmax(last_logits)
                last_preds = tf.to_int32(tf.multinomial(z, 1, seed=None, name=None))
                return last_preds

            one_constant = tf.constant(np.array([[0]]), dtype=tf.int64)
            last_preds = tf.cond(tf.equal(sampler_result, one_constant)[0][0], teacher_force, inference)

            preds = tf.concat((preds, last_preds), axis=1)  # [batch_size, i]
            logits = tf.concat((logits, last_logits[:, None, :]), axis=1)  # [batch_size, i, symbol_num]

            return i, preds, logits

        i, preds, logits = tf.while_loop(cond=not_finished,
                                         body=step,
                                         loop_vars=[0, preds, logits],
                                         shape_invariants=[
                                             tf.TensorShape([]),
                                             tf.TensorShape([None, None]),
                                             tf.TensorShape([None, None, None])],
                                         back_prop=True if is_training else False)
        # preds = preds[:, 1:]
        # preds = tf.concat((preds, tf.zeros(shape=[batch_size, self._hp.maxlen - y_max_length], dtype=tf.int32)), axis=1)
        logits = tf.concat((logits, tf.zeros(shape=[batch_size, self._hp.maxlen - y_max_length, self._data.symbol_num])),
                           axis=1)
        preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
        return preds, logits

    def beam_search(self, encoder_output, reuse):
        """Beam search in graph."""
        beam_size, batch_size = self._hp.test.beam_size, tf.shape(encoder_output)[0]
        inf = 1e10

        def get_bias_scores(scores, bias):
            """
            If a sequence is finished, we only allow one alive branch. This function aims to give one branch a zero score
            and the rest -inf score.
            Args:
                scores: A real value array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A real value array with shape [batch_size * beam_size, beam_size].
            """
            bias = tf.to_float(bias)
            b = tf.constant([0.0] + [-inf] * (beam_size - 1))
            b = tf.tile(b[None, :], multiples=[batch_size * beam_size, 1])
            return scores * (1 - bias[:, None]) + b * bias[:, None]

        def get_bias_preds(preds, bias):
            """
            If a sequence is finished, all of its branch should be </S> (3).
            Args:
                preds: A int array with shape [batch_size * beam_size, beam_size].
                bias: A bool array with shape [batch_size * beam_size].

            Returns:
                A int array with shape [batch_size * beam_size].
            """
            bias = tf.to_int32(bias)
            return preds * (1 - bias[:, None]) + bias[:, None] * 3

        # Prepare beam search inputs.
        # [batch_size, 1, *, hidden_units]
        encoder_output = encoder_output[:, None, :, :]
        # [batch_size, beam_size, feat_len, hidden_units]
        encoder_output = tf.tile(encoder_output, multiples=[1, beam_size, 1, 1])
        # [batch_size * beam_size, feat_len, hidden_units]
        encoder_output = tf.reshape(encoder_output, [batch_size * beam_size, -1, encoder_output.get_shape()[-1].value])
        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        preds = tf.ones([batch_size * beam_size, 1], dtype=tf.int32) * 2
        scores = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores = tf.tile(scores, multiples=[batch_size])  # [batch_size * beam_size]
        bias = tf.zeros_like(scores, dtype=tf.bool)  # 是否结束的标识位
        # 缓存的历史结果，[batch_size * beam_size, 0, num_blocks , hidden_units ]
        cache = tf.zeros([batch_size * beam_size, 0, self._hp.num_blocks, self._hp.hidden_units])

        def step(i, bias, preds, scores, cache):
            # Where are we.
            i += 1

            # Call decoder and get predictions.
            # [batch_size * beam_size, step, hidden_size]
            decoder_output = self._transformer_decoder(encoder_output, preds, is_training=False, reuse=reuse)
            last_preds, last_k_preds, last_k_scores = self.test_output(decoder_output, reuse=reuse)

            last_k_preds = get_bias_preds(last_k_preds, bias)
            last_k_scores = get_bias_scores(last_k_scores, bias)

            # Update scores.
            scores = scores[:, None] + last_k_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size ** 2])  # [batch_size, beam_size * beam_size]

            # Pruning.
            scores, k_indices = tf.nn.top_k(scores, k=beam_size)
            scores = tf.reshape(scores, shape=[-1])  # [batch_size * beam_size]
            base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])
            base_indices *= beam_size ** 2
            k_indices = base_indices + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]

            # Update predictions.
            last_k_preds = tf.gather(tf.reshape(last_k_preds, shape=[-1]), indices=k_indices)
            preds = tf.gather(preds, indices=k_indices / beam_size)
            # cache = tf.gather(cache, indices=k_indices / beam_size)
            preds = tf.concat((preds, last_k_preds[:, None]), axis=1)  # [batch_size * beam_size, i]

            # Whether sequences finished.
            bias = tf.equal(preds[:, -1], 3)  # </S>?

            return i, bias, preds, scores, cache

        def not_finished(i, bias, preds, scores, cache):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(bias)),
                tf.less_equal(
                    i,
                    tf.reduce_min([tf.shape(encoder_output)[1] + 50, self._hp.test.max_target_length])
                )
            )

        i, bias, preds, scores, cache = tf.while_loop(cond=not_finished,
                                                      body=step,
                                                      loop_vars=[0, bias, preds, scores, cache],
                                                      shape_invariants=[
                                                          tf.TensorShape([]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None, None]),
                                                          tf.TensorShape([None]),
                                                          tf.TensorShape([None, None, None, None])],
                                                      back_prop=False)

        scores = tf.reshape(scores, shape=[batch_size, beam_size])
        preds = tf.reshape(preds, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
        lengths = tf.reduce_sum(tf.to_float(tf.not_equal(preds, 3)), axis=-1)  # [batch_size, beam_size]
        lp = tf.pow((5 + lengths) / (5 + 1), self._hp.test.lp_alpha)  # Length penalty
        scores /= lp  # following GNMT
        max_indices = tf.to_int32(tf.argmax(scores, axis=-1))  # [batch_size]
        max_indices += tf.range(batch_size) * beam_size
        preds = tf.reshape(preds, shape=[batch_size * beam_size, -1])

        final_preds = tf.gather(preds, indices=max_indices)
        final_preds = final_preds[:, 1:]  # remove <S> flag
        return final_preds

    def test_output(self, decoder_output, reuse, k=None):
        """During test, we only need the last prediction at each time."""
        with tf.variable_scope("logits", reuse=reuse):
            # Final linear projection
            last_logits = tf.layers.dense(decoder_output[:, -1], self._data.symbol_num, reuse=reuse)

        if k is None:
            k = self._hp.test.beam_size
        last_preds = tf.to_int32(tf.arg_max(last_logits, dimension=-1))
        z = tf.nn.log_softmax(last_logits)
        last_k_scores, last_k_preds = tf.nn.top_k(z, k=k, sorted=False)
        last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores


class ExtraTransformer(Transformer):
    def __init__(self, hp, num_gpu, data):
        super(ExtraTransformer, self).__init__(hp, num_gpu, data)
        self.placeholders_signal()

    def placeholders_signal(self):
        with self.graph.as_default():
            src_pls = []
            for i, device in enumerate(self._devices):
                with tf.device(device):
                    pls_batch_x = tf.placeholder(dtype=tf.float32,
                                                 shape=[None, self._hp.audio_signal_length],
                                                 name='signal_pls_{}'.format(i))  # [batch, singal]
                    src_pls.append(pls_batch_x)
            self.signal_pls = tuple(src_pls)

    def build_model(self, is_training=True, get_wer=False):
        self.mean_loss, self.train_op, self.preds, self.istarget = self.get_train_op(is_training, get_wer=get_wer)

    def mel_spectrograms_feat(self, pcm):
        # A 512-point STFT with frames of 64 ms and 75% overlap.
        pcm_batch_size = tf.shape(pcm)[0]
        stfts = tf.contrib.signal.stft(pcm, frame_length=400, frame_step=160,
                                       fft_length=512)
        spectrograms = tf.abs(stfts)

        # Warp the linear scale spectrograms into the mel-scale.
        num_spectrogram_bins = stfts.shape[-1].value
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = 0.0, 8000.0, 80
        linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, self._hp.sample_rate, lower_edge_hertz,
            upper_edge_hertz)
        mel_spectrograms = tf.tensordot(
            spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
            linear_to_mel_weight_matrix.shape[-1:]))

        # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
        log_mel_spectrograms = tf.log(mel_spectrograms + 1e-6)

        # stack-4_downdownsampled-3
        log_mel_spectrograms_reshape = tf.reshape(log_mel_spectrograms[:, :self._hp.audio_length * 3, :],
                                                  [-1, self._hp.audio_length, 3 * 80])

        downdownsampled_index = tf.range(3, self._hp.log_mel_length, 3)
        downdownsampled_index = tf.reshape(downdownsampled_index, [1, self._hp.audio_length])
        downdownsampled_index = tf.tile(downdownsampled_index, [pcm_batch_size, 1])

        batch_index = tf.reshape(tf.range(pcm_batch_size), [-1, 1])
        batch_index = tf.tile(batch_index, [1, self._hp.audio_length])

        downdownsampled_index_nd_stack = tf.stack([batch_index, downdownsampled_index], axis=2)

        downdownsampled_values = tf.gather_nd(log_mel_spectrograms, downdownsampled_index_nd_stack)
        log_mel_spectrograms_reshape = tf.concat([log_mel_spectrograms_reshape, downdownsampled_values], 2)
        return log_mel_spectrograms_reshape

    def make_parallel(self, reuse=None, is_training=True, get_wer=False):
        if len(self.src_pls) <= 1:
            raise ValueError

        with self.graph.as_default():
            loss_list, gv_list = [], []
            preds_list, istarget_list = [], []
            cache = {}
            load = dict([(d, 0) for d in self._devices])
            for i, (X, Y, device) in enumerate(zip(self.signal_pls, self.dst_pls, self._devices)):

                def daisy_chain_getter(getter, name, *args, **kwargs):
                    """Get a variable and cache in a daisy chain."""
                    device_var_key = (device, name)
                    if device_var_key in cache:
                        # if we have the variable on the correct device, return it.
                        return cache[device_var_key]
                    if name in cache and "decoder" not in name and "logits" not in name:
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
                                       # custom_getter=daisy_chain_getter,
                                       reuse=reuse):
                    # with tf.device(device_setter):
                    with tf.device(device):
                        X = self.mel_spectrograms_feat(X)
                        if not self._hp.train.is_scheduled:
                            loss, preds, istarget = self.model_loss(X, Y, reuse=reuse, is_training=is_training,
                                                                    scope="transformer")
                        else:
                            loss, preds, istarget = self.model_scheduled_sampling_output(X, Y, reuse=reuse,
                                                                                         is_training=is_training,
                                                                                         scope="transformer")
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
            for i, (X, Y, device) in enumerate(zip(self.signal_pls, self.dst_pls, self._devices)):
                if i > 0:
                    reuse = True
                with tf.variable_scope(tf.get_variable_scope(), reuse=reuse, initializer=self._initializer):
                    with tf.device(device):
                        X = self.mel_spectrograms_feat(X)
                        prediction = self.model_beam_search_preds(X, reuse, is_training, scope="transformer")
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
