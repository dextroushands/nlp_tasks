import gensim
import os

from data_processor.tokenizer import tokenizer
import numpy as np
import h5py
import logging
logger = logging.getLogger(__name__)

class embedding(tokenizer):
    '''
    文本向量化
    '''
    def __init__(self, embedding_config):
        self.config = embedding_config
        super(embedding, self).__init__(embedding_config)

    def load_word2vec_model(self):
        '''
        加载word2vec模型
        :return:
        '''
        model_path = self.config.get('word2vec_path')
        if not os.path.exists(model_path):
            raise Exception("model_path did not exit, please check path")
        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
        return model

    def get_word_vectors(self, tokens):
        '''
        获取词向量
        :param tokens:
        :return:
        '''
        features = []
        embedding_size = self.config['embedding_size']
        word_vectors = np.zeros(embedding_size).tolist()
        model = self.load_word2vec_model()
        for word in tokens:
            if word in model.index2word:
                features.append(model.word_vec(word))
            else:
                features.append(word_vectors)
                print("{} is not in vocabulary!".format(word))
        # print(features)
        return features

    def save_vectors(self, vectors, name):
        '''
        保存向量到文件中
        :param vectors:
        :return:
        '''
        file_path = os.path.join(self.config['output_path'], name + '.npy')
        np.save(file_path, vectors)

    def get_one_hot_vectors(self, tokens):
        '''
        获取one-hot向量
        :param tokens:
        :return:
        '''
        raise NotImplemented

    def get_tf_idf_vectors(self, tokens):
        '''
        获取tf-idf向量
        :param tokens:
        :return:
        '''
        raise NotImplemented



