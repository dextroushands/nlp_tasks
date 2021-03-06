import os
import tensorflow as tf


class BaseModel(tf.keras.models):
    '''
    模型的基类
    '''
    def __init__(self, config):
        self.config = config
        super(BaseModel, self).__init__()

    def build_model(self):
        '''
        创建模型
        :return:
        '''
        raise NotImplemented

    def build_inputs(self):
        '''
        创建输入
        :return:
        '''
        raise NotImplemented

    def build_losses(self, labels, model_outputs, aux_losses) -> tf.Tensor:
        '''
        计算loss值
        :param labels:
        :param model_outputs:
        :param metrics:
        :return:
        '''
        raise NotImplemented

    def build_metrics(self, training: bool = True):
        """
        获取模型训练/验证的评价指标
        :param training:
        :return:
        """
        del training
        return []

    def process_metrics(self, metrics, labels, model_outputs):
        '''
        处理并更新评价指标
        :param metrics:
        :param labels:
        :param model_outputs:
        :return:
        '''
        for metric in metrics:
            metric.update_state(labels, model_outputs)

    def process_compiled_metrics(self, compiled_metrics, labels, model_outputs):
        '''
        处理并更新compiled metrics
        :param compiled_metrics:
        :param labels:
        :param model_outputs:
        :return:
        '''
        compiled_metrics.update_state(labels, model_outputs)

    def get_optimizer(self):
        '''
        选择优化算法
        :return:
        '''
        option = self.config['optimizer']
        optimizer = None
        learning_rate = self.config['learning_rate']
        if option == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate)
        if option == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate)
        if option == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate)
        return optimizer


    def train_op(self, loss, model):
        '''
        训练模型
        :param inputs:
        :return:
        '''
        optimizer = self.get_optimizer()
        with tf.GradientTape() as t:
            grads = t.gradients(loss, model.trainable_variables)
        # 对梯度进行梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(grads, self.config["max_grad_norm"])
        optimizer.apply_gradients(zip(clip_gradients, model.trainable_variables))
        tf.summary.scalar('loss', model.loss)
        tf.summary.scalar('lr', model.learning_rate)

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
            features, labels = inputs, inputs
        with tf.GradientTape() as tape:
            outputs = model(features, training=True)
            loss = self.build_losses(labels, outputs, aux_losses=None)

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
        optimizer.apply_gradients(list(zip(grads, tvars)))
        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
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
            features, labels = inputs, inputs

        outputs = self.infrence_step(features, model)
        loss = self.build_losses(labels, outputs, aux_losses=model.losses)

        logs = {self.loss: loss}
        if metrics:
            self.process_metrics(metrics, labels, outputs)
        if model.compiled_metrics:
            self.process_compiled_metrics(model.compiled_metrics, labels, outputs)
            logs.update({m.name: m.result() for m in metrics or []})
            logs.update({m.name: m.result() for m in model.metrics})
        return logs

    def inference_step(self, inputs, model:tf.keras.Model):
        '''
        模型推理
        :param inputs:
        :param model:
        :return:
        '''
        return model(inputs, training=False)

    def get_predictions(self, logits):
        '''
        模型预测结果
        :param input:
        :param model:
        :return:
        '''

        predictions = tf.argmax(logits, axis=-1, name='predictions')
        return predictions

    def save_ckpt_model(self, model):
        '''
        将模型保存成ckpt格式
        :param model:
        :return:
        '''
        save_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())),
                                 self.config["ckpt_model_path"])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_save_path = os.path.join(save_path, self.config["model_name"])

        checkpoint = tf.train.Checkpoint(model)
        checkpoint.save(model_save_path + '/model.ckpt')

    def save_pb_model(self, model):
        '''
        将模型保存成pb格式
        :param model:
        :return:
        '''
        raise NotImplemented

    def load_model(self):
        '''
        加载模型
        :return:
        '''
        raise NotImplemented









