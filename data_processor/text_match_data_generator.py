from data_processor.embedding import embedding
import numpy as np
import pandas as pd
import pickle
import os
from random import shuffle

class TextMatchDataGenerator(embedding):
    '''
    生成训练数据
    '''
    def __init__(self, config):
        super(TextMatchDataGenerator, self).__init__(config)
        self.config = config
        self.batch_size = config['batch_size']
        self.load_data()
        self.train_data, self.train_label, self.eval_data, self.eval_label = self.train_eval_split(self.query_idx,self.sim_query_idx, self.labels_idx, 0.2)

    def read_data(self, file_path, data_size=100):
        '''
        加载训练数据
        '''
        df = pd.read_csv(file_path)
        # query = [jieba.lcut(i) for i in df['sentence1'].values[0:data_size]]
        # sim = [jieba.lcut(i) for i in df['sentence2'].values[0:data_size]]
        query = [list(i) for i in df['sentence1'].values[0:data_size]]
        sim = [list(i) for i in df['sentence2'].values[0:data_size]]
        label = df['label'].values[0:data_size]

        return query, sim, label

    def save_input_tokens(self, query, sim, labels, word_to_index, label_to_index):
        '''
        保存处理完成的输入tokens，方便后续加载
        :param texts:
        :return:
        '''

        query_ids = []
        sim_query_ids = []
        label_ids = []
        for i in range(len(query)):
            query_tokens = self.tokens_to_ids(query[i], word_to_index)
            query_tokens = self.padding(query_tokens)
            query_ids.append(query_tokens)
            sim_query_tokens = self.tokens_to_ids(sim[i], word_to_index)
            sim_query_tokens = self.padding(sim_query_tokens)
            sim_query_ids.append(sim_query_tokens)
            label_id = self.labels_to_ids([labels[i]], label_to_index)
            label_ids.append(label_id)
        input_tokens = dict(query_idx=query_ids, sim_query_idx=sim_query_ids, labels_idx=label_ids)
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
        return query_ids, sim_query_ids, label_ids

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

            self.query_idx,self.sim_query_idx, self.labels_idx = np.array(train_data["query_idx"]), np.array(train_data["sim_query_idx"]), np.array(train_data["labels_idx"])
            self.vocab = self.word_to_index.keys()
            self.vocab_size = len(self.vocab)
        else:
            # 1，读取原始数据
            query, sim, labels = self.read_data(self.config['data_path'])
            print("read finished")

            inputs = query + sim
            all_words = self.get_all_words(inputs)
            word_to_index = self.word_to_index(all_words)
            label_to_index = self.label_to_index(labels)

            query_ids, sim_query_ids, label_ids = self.save_input_tokens(query, sim, labels, word_to_index, label_to_index)
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
            self.query_idx, self.sim_query_idx, self.labels_idx = query_ids, sim_query_ids, label_ids

    def train_eval_split(self, query_ids, sim_ids, labels, rate):

        split_index = int(len(query_ids) * rate)
        train_data = (query_ids[split_index:], sim_ids[split_index:])
        train_label = labels[split_index:]
        eval_data = (query_ids[:split_index], sim_ids[:split_index])
        eval_label = labels[:split_index]

        return train_data, train_label, eval_data, eval_label

    def gen_data(self, inputs_idx, labels_idx):
        '''
        生成批次数据
        :return:
        '''
        query_idx, sim_query_idx = inputs_idx[0], inputs_idx[1]
        batch_x, batch_y, batch_output_ids = [], [], []

        for i in range(len(query_idx)):
            query_token_ids = query_idx[i]
            sim_query_token_ids = sim_query_idx[i]
            target_ids = labels_idx[i]
            batch_x.append(query_token_ids)
            batch_y.append(sim_query_token_ids)
            batch_output_ids.append(target_ids)

            if len(batch_x) == self.batch_size:
                yield dict(
                    input_x_ids = np.array(batch_x, dtype="int64"),
                    input_y_ids=np.array(batch_y, dtype="int64"),
                    input_target_ids = np.array(batch_output_ids, dtype="float32")
                    # input_word_ids = batch_token_ids,
                    # input_target_ids = batch_output_ids
                )
                batch_x, batch_y, batch_output_ids = [], [], []

