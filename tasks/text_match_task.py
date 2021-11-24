import tensorflow as tf
from tensorflow_addons.layers import crf
from data_processor.text_match_data_generator import TextMatchDataGenerator


import json
import os
from keras_models.dnn_dssm import DSSM
from trainer.train_base import TrainBase

class TextMatchTask(TrainBase):
    '''
    构建文本匹配任务对象
    '''
    def __init__(self, task_config):
        self.config = task_config
        self.loss = "loss"

        super(TextMatchTask, self).__init__(task_config)
        self.data_generator = TextMatchDataGenerator(task_config)
        self.vocab_size = self.data_generator.vocab_size
        self.word_vectors = self.data_generator.word_vectors


    def build_model(self, vocab_size, word_vecotrs):
        '''
        构建ner模型
        :param vocab_size:
        :param word_vecotrs:
        :return:
        '''
        if self.config['model_name'] == 'dnn_dssm':

            return DSSM(self.config, vocab_size, word_vecotrs)
        else:
            raise Exception("please choose model")

    def build_inputs(self, inputs):
        '''
        输入转tensor
        :param inputs:
        :return:
        '''
        train_input = {
            "input_x_ids": tf.convert_to_tensor(inputs['input_x_ids']),
            "input_y_ids": tf.convert_to_tensor(inputs['input_y_ids']),
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

        with tf.name_scope('TextMatchTask/losses'):

            metrics = dict([(metric.name, metric) for metric in metrics])
            losses = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                                     tf.cast(model_outputs['logits'], tf.float32),
                                                                     from_logits=True)

            loss = tf.reduce_mean(losses)

            return loss

    def train_step(self,
                   inputs,
                   model:tf.keras.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   metrics=None):
        '''
        进行训练，前向和后向计算
        :param inputs:
        :param model:
        :param optimizer:
        :param metrics:
        :return:
        '''
        if isinstance(inputs, tuple) and len(inputs) == 2:
            features, labels = inputs
        else:
            features, labels = [inputs['input_x_ids'], inputs['input_y_ids']], inputs['input_target_ids']
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            loss = self.build_losses(labels, outputs, metrics, aux_losses=None)

            scaled_loss = loss / tf.distribute.get_strategy().num_replicas_in_sync
            #对于混合精度，该优化器对loss进行缩放防止数值下溢
            if isinstance(optimizer,
                          tf.keras.mixed_precision.experimental.LossScaleOptimizer):
                scaled_loss = optimizer.get_scaled_loss(scaled_loss)
        tvars = model.trainable_variables
        grads = tape.gradient(scaled_loss, tvars)

        if isinstance(optimizer,
                      tf.keras.mixed_precision.experimental.LossScaleOptimizer):
            grads = optimizer.get_unscaled_gradients(grads)
        optimizer.apply_gradients((grad, var) for (grad, var) in zip(grads, tvars) if grad is not None)
        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs['logits'])
            logs.update({m.name: m.result() for m in model.metrics})
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs['logits'])
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs


    def validation_step(self, inputs, model:tf.keras.Model, metrics=None):
        '''
        验证集验证模型
        :param input:
        :param model:
        :return:
        '''
        if isinstance(inputs, tuple) and len(inputs) == 2:
            features, labels = inputs
        else:
            features, labels = [inputs['input_x_ids'], inputs['input_y_ids']], inputs['input_target_ids']

        outputs = self.inference_step(features, model)
        loss = self.build_losses(labels, outputs, metrics, aux_losses=model.losses)

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs['logits'])
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs['logits'])
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def build_metrics(self, training=None):
        '''
        构建评价指标
        :param training:
        :return:
        '''
        # del training
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='text_match_metrics')
        ]

        # metrics = dict([(metric.name, metric) for metric in metrics])

        return metrics

if __name__=='__main__':
    with open("../model_configs/dnn_dssm.json", 'r') as fr:
        config = json.load(fr)
    print(config)
    text_match = TextMatchTask(config)
    vocab_size = text_match.vocab_size
    word_vectors = text_match.word_vectors
    print(len(word_vectors))
    model = text_match.build_model(vocab_size, word_vectors)
    text_match.train(model)