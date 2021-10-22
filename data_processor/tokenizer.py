'''
文本转化成tokens
'''
from data_processor.base_processor import data_base
from itertools import chain
import numpy as np
import pickle
import os

class tokenizer(data_base):
    '''
    文本转tokens
    '''
    def __init__(self, token_configs):
        self.token_configs = token_configs
        super(tokenizer, self).__init__(token_configs)

    def tokens_to_ids(self, tokens, tokens_to_index):
        '''
        token转索引
        :param tokens:
        :return:
        '''
        ids = [tokens_to_index.get(token, 1) for token in tokens]
        return ids

    def labels_to_ids(self, labels, labels_to_index):
        '''
        token转索引
        :param tokens:
        :return:
        '''
        ids = [labels_to_index.get(token) for token in labels]
        return ids

    def seq_labels_to_ids(self, labels, labels_to_index):
        '''
        token转索引
        :param tokens:
        :return:
        '''
        if len(labels) < self.config['seq_len']:
            labels += ['O'] * (self.config['seq_len'] - len(labels))
        else:
            labels = labels[:self.config['seq_len']]
        nan_id = labels_to_index.get('O')
        ids = [labels_to_index.get(token, nan_id) for token in labels]
        return ids

    def ids_to_tokens(self, ids, tokens_to_index):
        '''
        索引转成token
        :param ids:
        :return:
        '''
        tokens = [list(tokens_to_index.keys())[id] for id in ids]
        return tokens

    def multi_label_to_index(self, labels, label_to_index):
        '''
        多标签数据转索引
        :param labels:
        :return:
        '''
        label_idxs = np.zeros((len(labels), len(label_to_index)))

        for i, label in enumerate(labels):
            for l in label:
                id = label_to_index.get(l)
                label_idxs[i, id] = 1
        return label_idxs

    def word_to_index(self, all_words):
        '''
        生成词汇-索引字典
        :param texts:
        :return:
        '''

        #是否过滤低频词
        if self.config['freq_filter']:
            vocab = self.word_freq_filter(self.config['freq_filter'], all_words)
        else:
            vocab = self.get_vocab(all_words)
        #设置词典大小
        vocab = ["<PAD>", "<UNK>"] + vocab
        self.vocab_size = self.config['vocab_size']
        if len(vocab) < self.vocab_size:
            self.vocab_size = len(vocab)
        self.vocab = vocab[:self.vocab_size]
        #构建词典索引
        word_to_index = dict(zip(vocab, list(range(len(vocab)))))

        return word_to_index

    def label_to_index(self, labels):
        '''
        标签索引字典
        :param labels:
        :return:
        '''
        if not self.config['multi_label']:
            unique_labels = list(set(labels))  # 单标签转换
        else:
            unique_labels = list(set(chain(*labels)))#多标签转换
        label_to_index = dict(zip(unique_labels, list(range(len(unique_labels)))))
        return label_to_index

    def padding(self, tokens):
        '''
        将输入序列做定长处理
        :param tokens:
        :return:
        '''
        if len(tokens) < self.config['seq_len']:
            tokens += [0] * (self.config['seq_len'] - len(tokens))
        else:
            tokens = tokens[:self.config['seq_len']]
        return tokens

    def save_input_tokens(self, texts, labels, word_to_index, label_to_index):
        '''
        保存处理完成的输入tokens，方便后续加载
        :param texts:
        :return:
        '''

        input_ids = []
        label_ids = []
        for i in range(len(texts)):
            tokens = self.tokens_to_ids(texts[i], word_to_index)
            tokens = self.padding(tokens)
            input_ids.append(tokens)
            label_id = self.labels_to_ids([labels[i]], label_to_index)
            label_ids.append(label_id)
        input_tokens = dict(inputs_idx=input_ids, labels_idx=label_ids)
        if not os.path.exists(self.config['output_path']):
            os.mkdir(self.config['output_path'])
        #保存准备训练的tokens数据
        with open(os.path.join(self.config['output_path'], 'train_tokens.pkl'), "wb") as fw:
            pickle.dump(input_tokens, fw)
        # 保存预处理的word_to_index数据
        with open(os.path.join(self.config['output_path'], 'word_to_index.pkl'), "wb") as fw:
            pickle.dump(word_to_index, fw)
        # 保存预处理的word_to_index数据
        with open(os.path.join(self.config['output_path'], 'label_to_index.pkl'), "wb") as fw:
            pickle.dump(label_to_index, fw)
        return input_ids, label_ids