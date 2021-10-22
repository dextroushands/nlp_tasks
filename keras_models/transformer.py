import tensorflow as tf
# from official.nlp.transformer.transformer import Transformer
from official.nlp.modeling.layers import Transformer


class TransformerNet(tf.keras.Model):
    '''
    基于transformer的网络结构
    '''
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config

        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        # 定义输入
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_word_ids')

        # embedding层
        with tf.name_scope('embedding'):
            if not self.config['use_word2vec']:
                embedding_w = tf.keras.initializers.glorot_normal()(
                    shape=[self.vocab_size, self.config['emebdding_size']],
                    dtype=tf.float32)
            else:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, tf.float32), name='embedding_w')

        class GatherLayer(tf.keras.layers.Layer):
            def call(self, indices, params):
                return tf.gather(params, indices, name='embedded_words')

        # 利用词嵌入矩阵将输入数据转成词向量，shape=[batch_size, seq_len, embedding_size]
        embedded_words = GatherLayer()(word_ids, embedding_w)

        #构建transformer层
        trans_layer = Transformer(
            num_attention_heads=self.config['num_attention_heads'],
            intermediate_size=self.config['intermediate_size'],
            intermediate_activation=self.config['intermediate_activation'],
            dropout_rate=self.config['dropout_rate'],
            attention_dropout_rate=self.config['attention_dropout_rate'],
            output_range=None,
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        )
        #transformer的输出[batch_size, seq_len, hidden_size]，不改变输入的维度
        sequence_output = trans_layer(embedded_words)

        #将输出摊平成二维
        sequence_output = tf.reshape(sequence_output, shape=(self.config['batch_size'], self.config['seq_len']*self.config['embedding_size']))

        #全连接层,输出维度为[batch_size, num_classes]
        full_connect_layer = tf.keras.layers.Dense(self.config['num_classes'])(sequence_output)

        self.logits = full_connect_layer

        outputs = dict(logits=self.logits)

        super(TransformerNet, self).__init__(inputs=[word_ids], outputs=outputs)

