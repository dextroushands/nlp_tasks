import tensorflow as tf
from tensorflow_addons.layers import CRF


class BilstmCRF(tf.keras.Model):
    '''
    bilstm+crf实体识别网络结构
    '''
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        #定义输入
        word_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_word_ids')

        #embedding层
        class GatherLayer(tf.keras.layers.Layer):
            def __init__(self, config, vocab_size, word_vectors):
                super(GatherLayer, self).__init__()
                self.config = config

                self.vocab_size = vocab_size
                self.word_vectors = word_vectors

            def build(self, input_shape):
                with tf.name_scope('embedding'):
                    if not self.config['use_word2vec']:
                        self.embedding_w = tf.keras.initializers.glorot_normal()(
                            shape=[self.vocab_size, self.config['embedding_size']],
                            dtype=tf.float32)
                    else:
                        self.embedding_w = tf.Variable(tf.cast(self.word_vectors, tf.float32), trainable=True,
                                                       name='embedding_w')
                self.build = True

            def call(self, indices):
                return tf.gather(self.embedding_w, indices, name='embedded_words')

            def get_config(self):
                config = super(GatherLayer, self).get_config()

                return config

        # 利用词嵌入矩阵将输入数据转成词向量，shape=[batch_size, seq_len, embedding_size]
        embedded_words = GatherLayer(config, vocab_size, word_vectors)(word_ids)

        #bi-lstm层
        forward_layer = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'], return_sequences=True)
        backward_layer = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                              return_sequences=True, go_backwards=True)
        #双向网络层，得到的结果拼接，维度大小为[batch_size, forward_size+backward_size]
        lstm_res = tf.keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer)(embedded_words)

        #crf层
        decoded_sequence, potentials, sequence_length, chain_kernel = CRF(self.config['tag_categories'])(lstm_res)

        #输出为【batch_size, seq_len, num_tags]
        self.logits = potentials

        outputs = dict(logits=self.logits, decoded_outputs=decoded_sequence)

        super(BilstmCRF, self).__init__(inputs=[word_ids], outputs=outputs)







