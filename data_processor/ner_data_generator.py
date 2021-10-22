from data_processor.embedding import embedding
import numpy as np
import pandas as pd
import pickle
import os


class NERDataGenerator(embedding):
    '''
    生成训练数据
    '''
    def __init__(self, config):
        super(NERDataGenerator, self).__init__(config)
        self.config = config
        self.batch_size = config['batch_size']
        self.load_data()
        self.train_data, self.train_label, self.eval_data, self.eval_label = self.train_eval_split(self.inputs_idx, self.labels_idx, 0.2)

    def read_data(self, path):
        inputs = []
        labels = []
        with open(os.path.join(path, 'source_BIO_2014_cropus.txt'), 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                inputs.append(line.split(sep=' '))
        with open(os.path.join(path, 'target_BIO_2014_cropus.txt'), 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                labels.append(line.split(sep=' '))
        return inputs[:100], labels[:100]

    def get_labels(self):
        return ['O', 'B_LOC', 'I_LOC', 'B_PER', 'I_PER', 'B_ORG', 'I_ORG', 'B_T', 'I_T']

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
            label_id = self.seq_labels_to_ids(labels[i], label_to_index)
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

    def load_data(self):
        '''
        加载预处理好的数据
        :return:
        '''

        if os.path.exists(os.path.join(self.config['output_path'], "train_tokens.pkl")) and \
                os.path.exists(os.path.join(self.config['output_path'], "label_to_index.pkl")) and \
                os.path.exists(os.path.join(self.config['output_path'], "word_to_index.pkl")):
            print("load existed train data")
            with open(os.path.join(self.config['output_path'], "word_to_index.pkl"), "rb") as f:
                self.word_to_index = pickle.load(f)
            with open(os.path.join(self.config['output_path'], "label_to_index.pkl"), "rb") as f:
                self.label_to_index = pickle.load(f)
            with open(os.path.join(self.config['output_path'], "train_tokens.pkl"), "rb") as f:
                train_data = pickle.load(f)

            if os.path.exists(os.path.join(self.config['output_path'], "word_vectors.npy")):
                print("load word_vectors")
                self.word_vectors = np.load(os.path.join(self.config['output_path'], "word_vectors.npy"),
                                            allow_pickle=True)

            self.inputs_idx, self.labels_idx = np.array(train_data["inputs_idx"]), np.array(train_data["labels_idx"])
            self.vocab = self.word_to_index.keys()
            self.vocab_size = len(self.vocab)
        else:
            # 1，读取原始数据
            inputs, labels = self.read_data(self.config['data_path'])
            print("read finished")

            targets = self.get_labels()
            all_words = self.get_all_words(inputs)
            word_to_index = self.word_to_index(all_words)
            label_to_index = self.label_to_index(targets)

            inputs_idx, labels_idx = self.save_input_tokens(inputs, labels, word_to_index, label_to_index)
            print('text to tokens process finished')

            # # 2，得到去除低频词和停用词的词汇表
            # word_to_index, all_words = self.word_to_index(inputs)
            # print("word process finished")
            #
            # # 3，得到词汇表
            # label_to_index = self.label_to_index(labels)
            # print("vocab process finished")
            #
            # # 4，输入转索引
            # inputs_idx = [self.tokens_to_ids(text, word_to_index) for text in all_words]
            # print("index transform finished")
            #
            # # 5，对输入做padding
            # inputs_idx = self.padding(inputs_idx)
            # print("padding finished")
            #
            # # 6，标签转索引
            # labels_idx = self.tokens_to_ids(labels, label_to_index)
            # print("label index transform finished")

            # 7, 加载词向量
            if self.config['word2vec_path']:
                word_vectors = self.get_word_vectors(self.vocab)
                self.word_vectors = word_vectors
                # 将本项目的词向量保存起来
                self.save_vectors(self.word_vectors, 'word_vectors')

            # train_data = dict(inputs_idx=inputs_idx, labels_idx=labels_idx)
            # with open(os.path.join(self.config['output_path'], "train_data.pkl"), "wb") as fw:
            #     pickle.dump(train_data, fw)
            # labels_idx = labels
            self.inputs_idx, self.labels_idx = inputs_idx, labels_idx

    def train_eval_split(self, data, labels, rate):
        '''
        划分训练和验证集
        :param data:
        :param labels:
        :param rate:
        :return:
        '''
        np.random.shuffle(data)
        perm = int(len(data) * rate)
        train_data = data[perm:]
        eval_data = data[:perm]
        train_label = labels[perm:]
        eval_label = labels[:perm]
        return train_data, train_label, eval_data, eval_label


    def gen_data(self, inputs_idx, labels_idx):
        '''
        生成批次数据
        :return:
        '''
        batch_token_ids, batch_output_ids = [], []

        for i in range(len(inputs_idx)):
            token_ids = inputs_idx[i]
            target_ids = labels_idx[i]
            batch_token_ids.append(token_ids)
            batch_output_ids.append(target_ids)

            if len(batch_token_ids) == self.batch_size:
                yield dict(
                    input_word_ids = np.array(batch_token_ids, dtype="int64"),
                    input_target_ids = np.array(batch_output_ids, dtype="float32")
                    # input_word_ids = batch_token_ids,
                    # input_target_ids = batch_output_ids
                )
                batch_token_ids, batch_output_ids = [], []

