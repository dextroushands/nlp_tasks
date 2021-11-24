import tensorflow as tf
from tensorflow_addons.layers import crf

import json
import os
from keras_models.bilstm_crf import BilstmCRF
from trainer.train_base import TrainBase
from data_processor.classifier_data_generator import ClassifierDataGenerator
from data_processor.ner_data_generator import NERDataGenerator

class NERTask(TrainBase):
    '''
    构建ner任务对象
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = "loss"

        super(NERTask, self).__init__(task_config)
        self.data_generator = NERDataGenerator(task_config)
        self.vocab_size = self.data_generator.vocab_size
        self.word_vectors = self.data_generator.word_vectors

    def build_model(self, vocab_size, word_vecotrs):
        '''
        构建ner模型
        :param vocab_size:
        :param word_vecotrs:
        :return:
        '''
        if self.config['model_name'] == 'bilstm_crf':

            return BilstmCRF(self.config, vocab_size, word_vecotrs)
        else:
            raise Exception("please choose model")

    def build_inputs(self, inputs):
        '''
        输入转tensor
        :param inputs:
        :return:
        '''
        train_input = {
            "input_word_ids": tf.convert_to_tensor(inputs['input_word_ids']),
            "input_target_ids": tf.convert_to_tensor(inputs['input_target_ids'])
        }
        return train_input

    def build_losses(self, labels, model_outputs, metrics, aux_losses=None) -> tf.Tensor:
        '''
        构建ner问题损失函数
        :param labels:
        :param model_outputs:
        :param aux_losses:
        :return:
        '''

        with tf.name_scope('NERTask/losses'):

            metrics = dict([(metric.name, metric) for metric in metrics])
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                     tf.cast(model_outputs['logits'], tf.float32),
                                                                     from_logits=True)

            loss = tf.reduce_mean(losses)

            return loss

    def build_metrics(self, training=None):
        '''
        构建评价指标
        :param training:
        :return:
        '''
        # del training
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='ner_metrics')
        ]

        # metrics = dict([(metric.name, metric) for metric in metrics])

        return metrics

if __name__=='__main__':
    with open("../model_configs/bilstm_crf.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    ner = NERTask(config)
    vocab_size = ner.vocab_size
    word_vectors = ner.word_vectors
    print(len(word_vectors))
    model = ner.build_model(vocab_size, word_vectors)
    ner.train(model)