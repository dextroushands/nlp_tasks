import tensorflow as tf
import os
import json

from keras_models.model_base import BaseModel
from data_processor.base_processor import data_base
from data_processor.tokenizer import tokenizer
from keras_models.text_cnn import TextCnn
from trainer.train_base import TrainBase
from keras_models.transformer_encoder import TransformerNet
from data_processor.classifier_data_generator import ClassifierDataGenerator
from data_processor.ner_data_generator import NERDataGenerator


class ClassifyTask(TrainBase):
    '''
    文本分类任务
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = "loss"

        super(ClassifyTask, self).__init__(task_config)
        self.data_generator = ClassifierDataGenerator(task_config)
        self.vocab_size = self.data_generator.vocab_size
        self.word_vectors = self.data_generator.word_vectors

    def build_model(self, vocab_size, word_vecotrs):
        '''
        创建模型
        :return:
        '''
        if self.config['model_name'] == 'text_cnn':

            return TextCnn(self.config, vocab_size, word_vecotrs)

        elif self.config['model_name'] == 'transformer':

            return TransformerNet(self.config, vocab_size, word_vecotrs)
        else:
            raise Exception("please choose model")

    def build_inputs(self, input):
        '''
        将文本转成tensor
        :return:
        '''
        train_input = {
            "input_word_ids": tf.convert_to_tensor(input['input_word_ids']),
            "input_target_ids": tf.convert_to_tensor(input['input_target_ids'])
            }
        return train_input

    def build_losses(self, labels, model_outputs, metrics, aux_losses=None) -> tf.Tensor:
        '''
        构建分类问题损失函数
        :param labels:
        :param model_outputs:
        :param aux_losses:
        :return:
        '''

        with tf.name_scope('ClassifyTask/losses'):

            # metrics = dict([(metric.name, metric) for metric in metrics])
            if self.config['num_classes'] > 1:
                losses = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                         tf.cast(model_outputs['logits'], tf.float32),
                                                                         from_logits=True)
            else:
                losses = tf.keras.losses.categorical_crossentropy(labels,
                                                                  tf.cast(model_outputs['logits'], tf.float32),
                                                                  from_logits=True
                                                                  )
            # metrics['losses'].update_state(losses)
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
            tf.keras.metrics.SparseCategoricalAccuracy(name='classifier_metrics')
        ]

        # metrics = dict([(metric.name, metric) for metric in metrics])

        return metrics

    def check_exist_model(self, model):
        '''
        检查是否存在模型文件
        :return:
        '''
        # ckpt = tf.train.Checkpoint(model=model)
        init_checkpoint = os.path.join(self.config['ckpt_model_path'], self.config['model_name'])

        # ckpt.restore(init_checkpoint).assert_existing_objects_matched()
        model.load_weights(init_checkpoint).assert_existing_objects_matched()

if __name__=='__main__':
    with open("../model_configs/transformer.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    classifier = ClassifyTask(config)
    vocab_size = classifier.vocab_size
    word_vectors = classifier.word_vectors
    print(len(word_vectors))
    model = classifier.build_model(vocab_size, word_vectors)
    # if os.path.exists(config['ckpt_model_path']):
    #     classifier.check_exist_model(model)
    # classifier.train(model)
    classifier.fit_train(model)






