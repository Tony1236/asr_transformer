# -*- coding: utf-8 -*-
import os

from file_wav import *
import librosa

current_relative_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), x))


class DataSpeech(object):
    def __init__(self, data_type):
        self.data_type = data_type

        self.symbol_num = 0  # 记录拼音符号数量
        self.slash = "/"
        self.data_path = current_relative_path("../data/")
        self.list_symbol = self.get_symbol_list()

        self.data_num = 0
        self.data_set = ["speechling_label"]
        self.dic_wavlist = {}
        self.list_wavnum = []

        self.dic_symbollist = {}
        self.list_symbolnum = []

        self.load_data()

    def get_symbol_list(self):
        """
        加载拼音符号列表，用于标记符号
        返回一个列表list类型变量
        """
        # 打开文件并读入
        txt_obj = open(current_relative_path('../conf/word_index.txt'), 'r')
        txt_text = txt_obj.read()
        # 文本分割
        txt_lines = txt_text.split('\n')
        # 初始化符号列表
        list_symbol = ['<PAD>', '<UNK>', '<S>', '</S>']
        for i in txt_lines:
            if i:
                txt_l = i.strip().split('\t')
                list_symbol.append(txt_l[0].decode("utf-8"))
        txt_obj.close()
        self.symbol_num = len(list_symbol)
        return list_symbol

    def load_data(self):
        # 设定选取哪一项作为要使用的数据集
        if self.data_type == 'train':
            wav_suffix = "train.wav.lst"
            syllable_suffix = "train.txt"
        elif self.data_type == 'dev':
            wav_suffix = "dev.wav.lst"
            syllable_suffix = "dev.txt"
        else:
            wav_suffix = "test.wav.lst"
            syllable_suffix = "test.txt"

        for sub_data_set in self.data_set:
            filename_wavlist = self.data_path + self.slash + sub_data_set + self.slash + wav_suffix
            filename_symbollist = self.data_path + self.slash + "audio_language" + \
                                  self.slash + sub_data_set + self.slash + syllable_suffix

            sub_dic_wavlist, sub_list_wavnum = get_wav_list(filename_wavlist)

            self.list_wavnum += sub_list_wavnum
            self.dic_wavlist = dict(self.dic_wavlist, **sub_dic_wavlist)

            sub_dic_symbollist, sub_list_symbolnum = get_wav_word_symbol(filename_symbollist)
            self.list_symbolnum += sub_list_symbolnum
            self.dic_symbollist = dict(self.dic_symbollist, **sub_dic_symbollist)

        self.data_num = self.get_data_num()

    def get_data_num(self):
        num_wavlist = len(self.dic_wavlist)
        num_symbollist = len(self.dic_symbollist)
        return num_wavlist

    def yield_batch_data(self, batch_size=32, audio_length=1600, maxlen=30, audio_feature_length=200, feat_type="fft",
                         if_stack_subsample=True, speed=False, return_file_name=False):
        data_num = self.get_data_num()
        start_num = 0
        random.shuffle(self.list_wavnum)
        while True:
            x = np.zeros((batch_size, audio_length, audio_feature_length), dtype=np.float)
            y = np.zeros((batch_size, maxlen), dtype=np.int64)  # 64 + /s
            files_name = []
            i = 0
            while i < batch_size:
                data_input, data_labels, file_name = self.get_data(start_num, feat_type=feat_type,
                                                                   if_stack_subsample=if_stack_subsample, speed=speed,
                                                                   return_file_name=return_file_name)
                if len(data_input) <= audio_length and len(data_labels) <= maxlen:
                    x[i, 0:len(data_input)] = data_input
                    y[i, 0:len(data_labels)] = data_labels
                    files_name.append(file_name)
                    i += 1
                start_num += 1
                if start_num >= data_num:
                    random.shuffle(self.list_wavnum)
                    start_num = 0
            if return_file_name:
                yield x, y, files_name
            else:
                yield x, y

    def yield_singal_batch_data(self, batch_size, singal_length, maxlen, speed=False, if_buf_to_float=False):
        data_num = self.get_data_num()
        start_num = 0
        random.shuffle(self.list_wavnum)
        while True:
            x = np.zeros((batch_size, singal_length), dtype=np.int16)
            y = np.zeros((batch_size, maxlen), dtype=np.int64)  # 64 + /s
            files_name = []
            i = 0
            while i < batch_size:
                try:
                    data_input, data_labels, file_name = self.get_wav_data(start_num, speed)
                    if len(data_input) <= singal_length and len(data_labels) <= maxlen:
                        if if_buf_to_float:
                            data_input = librosa.util.buf_to_float(data_input)
                        x[i, 0:len(data_input)] = data_input
                        y[i, 0:len(data_labels)] = data_labels
                        files_name.append(file_name)
                        i += 1
                except:
                    pass
                start_num += 1
                if start_num >= data_num:
                    random.shuffle(self.list_wavnum)
                    start_num = 0
            yield x, y, files_name

    def get_wav_data(self, n_start, speed=False):
        file_name_index = self.list_wavnum[n_start]
        filename = self.dic_wavlist[file_name_index]
        list_symbol = self.dic_symbollist[file_name_index]

        wavsignal, fs = read_wav_data(self.data_path + self.slash + filename, speed=speed)

        # 获取输出特征
        feat_out = []
        for i in list_symbol.decode("utf-8"):
            if i:
                n = self.symbol2num(i)
                feat_out.append(n)
        feat_out.append(3)
        return wavsignal[0], np.array(feat_out, dtype=np.int64), filename

    def get_wav_feat(self, singal_length, path):
        wavsignal, fs = read_wav_data(path, speed=False)
        x = np.zeros((1, singal_length), dtype=np.int16)
        x[0, 0:len(wavsignal[0])] = wavsignal[0]
        return x

    def get_data(self, n_start, feat_type="fft", if_stack_subsample=True, speed=False, return_file_name=False):
        file_name_index = self.list_wavnum[n_start]
        filename = self.dic_wavlist[file_name_index]
        list_symbol = self.dic_symbollist[file_name_index]

        wavsignal, fs = read_wav_data(self.data_path + self.slash + filename, speed=speed)

        # 获取输出特征
        feat_out = []
        for i in list_symbol:
            if i:
                n = self.symbol2num(i)
                feat_out.append(n)
        feat_out.append(3)

        # 获取输入特征
        if feat_type == "fft":
            data_input = get_frequency_feature(wavsignal, fs)
        elif feat_type == "mfcc":
            data_input = get_mfcc_feature(wavsignal, fs)
        elif feat_type == "log_fbank":
            data_input = get_log_mel_fbank(wavsignal, fs, if_stack_subsample=if_stack_subsample)
        else:
            raise ValueError
        data_label = np.array(feat_out)
        if not return_file_name:
            return data_input, data_label, None
        else:
            return data_input, data_label, filename

    def symbol2num(self, symbol):
        if symbol in self.list_symbol:
            return self.list_symbol.index(symbol)
        return self.list_symbol.index('<UNK>')

    def batch_seq2symbol(self, batch):
        symbol_list = []
        for b in batch:
            symbol_list.append(" ".join([self.num2symbol(i) for i in b if i not in (0, 3)]))
        return symbol_list

    def num2symbol(self, num):
        return self.list_symbol[num]
