# -*- coding: utf-8 -*-
class Train:
    learning_rate = 0.0001
    optimizer = 'adam'
    var_filter = ''
    grads_clip = 5
    loss_ce = 1.0
    loss_mwer = 0.5
    num_batch = 5614
    is_scheduled = False


class Test:
    beam_size = 10
    max_target_length = 34
    lp_alpha = 0.6


class Hyperparams:
    '''Hyperparameters'''
    # training
    batch_size = 8  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    maxlen = 34  # Maximum number of words in a sentence. alias = T.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 10
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    sample_rate = 16000
    audio_length = 532
    audio_feature_length = 320
    audio_signal_length = 16 * sample_rate
    log_mel_length = 1598

    save_epochs = 3
    beam_size = 10
    end_id = 3
    filter_size = 5
    num_filters_1 = 64
    num_filters_2 = 128
    filter_sizes = [1, 3, 5, 7]
    bn_decay = 0.9997
    bn_epsilon = 0.001

    train = Train
    test = Test

    loss_type = ['ce', 'l2', 'mwer']