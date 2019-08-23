# -*- coding: utf-8 -*-
import wave

import numpy as np
import librosa
from python_speech_features import mfcc, delta, logfbank
from scipy.fftpack import fft
import math
import random
import cv2


def get_wav_list(filename):
    """
    读取一个wav文件列表，返回一个存储该列表的字典类型值
    ps:在数据中专门有几个文件用于存放用于训练、验证和测试的wav文件列表
    """
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_filelist = {}  # 初始化字典
    list_wavmark = []  # 初始化wav列表
    for i in txt_lines:
        if i:
            txt_l = i.strip().split(' ')
            dic_filelist[txt_l[0]] = " ".join(txt_l[1:])
            list_wavmark.append(txt_l[0])
    txt_obj.close()
    return dic_filelist, list_wavmark


def get_wav_symbol(filename):
    """
    读取指定数据集中，所有wav文件对应的语音符号
    返回一个存储符号集的字典类型值
    """
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_symbol_list = {}  # 初始化字典
    list_symbolmark = []  # 初始化symbol列表
    for i in txt_lines:
        if i:
            txt_l = i.strip().split(' ')
            dic_symbol_list[txt_l[0]] = txt_l[1:]
            list_symbolmark.append(txt_l[0])
    txt_obj.close()
    return dic_symbol_list, list_symbolmark


def get_wav_word_symbol(filename):
    txt_obj = open(filename, 'r')  # 打开文件并读入
    txt_text = txt_obj.read()
    txt_lines = txt_text.split('\n')  # 文本分割
    dic_symbol_list = {}  # 初始化字典
    list_symbolmark = []  # 初始化symbol列表
    for i in txt_lines:
        if i:
            txt_l = i.strip().split('[space]')
            dic_symbol_list[txt_l[0]] = txt_l[1]
            list_symbolmark.append(txt_l[0])
    txt_obj.close()
    return dic_symbol_list, list_symbolmark


def read_wav_data(filename, speed=False):
    """
    读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
    """
    wav = wave.open(filename, "rb")  # 打开一个wav格式的声音文件流
    num_frame = wav.getnframes()  # 获取帧数
    num_channel = wav.getnchannels()  # 获取声道数
    framerate = wav.getframerate()  # 获取采样率
    str_data = wav.readframes(num_frame)  # 读取全部的帧
    # print num_channel, framerate, wav.getsampwidth()
    wav.close()  # 关闭流
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel  # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T  # 将矩阵转置
    if speed and wave_data.shape[0] == 1:
        if random.random() <= 0.5:
            speed_rate = random.random() * 0.2 + 0.9
            wave_data = [speed_tune(wave_data[0], speed_rate)]
    return wave_data, framerate


def read_wav_data_from_librosa(filename, framerate=16000, speed=False):
    wave_data = librosa.load(filename, sr=framerate, mono=True)[0]
    if speed:
        if random.random() <= 0.5:
            speed_rate = random.random() * 0.2 + 0.9
            wave_data = speed_tune(wave_data, speed_rate)
    return [wave_data], framerate

x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗


def get_frequency_feature(wavsignal, fs):
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    window_length = fs / 1000 * time_window  # 计算窗长度的公式，目前全部为400固定值

    wav_arr = np.array(wavsignal)
    wav_length = wav_arr.shape[1]

    range0_end = int(len(wavsignal[0]) * 1.0 / fs * 1000 - time_window) // 10  # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据

    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400

        data_line = wav_arr[0, p_start:p_end]

        data_line = data_line * w  # 加窗

        data_line = np.abs(fft(data_line)) / wav_length

        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

    data_input = np.log(data_input + 1)
    return data_input


def get_mfcc_feature(wavsignal, fs):
    # 获取输入特征
    feat_mfcc = mfcc(wavsignal[0], fs)
    feat_mfcc_d = delta(feat_mfcc, 2)
    feat_mfcc_dd = delta(feat_mfcc_d, 2)
    # 返回值分别是mfcc特征向量的矩阵及其一阶差分和二阶差分矩阵
    wav_feature = np.column_stack((feat_mfcc, feat_mfcc_d, feat_mfcc_dd))
    return wav_feature


def get_log_mel_fbank(wavsignal, fs, add_delta=False, nor=False, if_stack_subsample=True):
    fbank_feat = logfbank(wavsignal[0], fs, nfilt=80)
    if add_delta:
        fbank_feat = np.column_stack((fbank_feat, delta(fbank_feat, 2)))
    if nor:
        fbank_feat = wav_scale(fbank_feat)
    if if_stack_subsample:
        fbank_feat = stack_subsample(fbank_feat)
    return fbank_feat


def stack_subsample(wav_feat, stack_num=4, frame_rate=3):
    # stacked with 3 frames to the left and downsampled to a 30ms frame rate
    new_feat = []
    length = wav_feat.shape[0]
    for i in range(stack_num - 1, length, frame_rate):
        stack_feat = wav_feat[i + 1 - stack_num: i + 1].flatten()
        new_feat.append(stack_feat)

    if (length - stack_num) % frame_rate != 0:
        new_feat.append(wav_feat[length - stack_num: length].flatten())
    return np.array(new_feat)


def wav_scale(energy):
    '''
    语音信号能量归一化
    '''
    means = energy.mean()  # 均值
    var = energy.var()  # 方差
    e = (energy - means) / math.sqrt(var)  # 归一化能量
    return e


def speed_tune(wav, speed_rate):
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()
    return wav_speed_tune


if __name__ == '__main__':
    wavsignal, fs = read_wav_data("../../data/forvo/搭讪_来这走走怎么样？_126925.wav")
    print wavsignal[0].max(), wavsignal[0].min(), wavsignal.shape
    import librosa
    signal = librosa.util.buf_to_float(wavsignal[0], dtype=np.float32)
    print signal.shape, signal.max(), signal.min()
    # wavsignal, fs = read_wav_data_from_librosa("../../data/forvo/3_一二三四五六七八九十_4297085.wav")
    # print wavsignal[0].max(), wavsignal[0].min(), wavsignal[0].shape
    # import time
    #
    # s = time.time()
    # for _ in range(100):
    #     wav = read_wav_data_from_librosa("../../data/forvo/搭讪_来这走走怎么样？_126925.wav", speed=True)
    #
    # print (time.time() - s) / 100 * 64