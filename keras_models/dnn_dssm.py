import tensorflow as tf

class DSSM(tf.keras.Model):
    '''
    dssm模型
    '''
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        # 定义输入
        query = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_x_ids')
        sim_query = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_y_ids')

        # embedding层
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
        query_embedding = GatherLayer(config, vocab_size, word_vectors)(query)
        query_embedding = tf.reshape(query_embedding, [self.config['batch_size'], self.config['seq_len']*self.config['embedding_size']])
        sim_query_embedding = GatherLayer(config, vocab_size, word_vectors)(sim_query)
        sim_query_embedding = tf.reshape(sim_query_embedding, [self.config['batch_size'], self.config['seq_len']*self.config['embedding_size']])

        #3层dnn提取特征[300,300,128], 输出为[batch_size, dense_embedding]
        for size in self.config['hidden_size']:
            query_embedding = tf.keras.layers.Dense(size)(query_embedding)
            sim_query_embedding = tf.keras.layers.Dense(size)(sim_query_embedding)

        #than激活函数
        query_embedding = tf.keras.activations.tanh(query_embedding)
        sim_query_embedding = tf.keras.activations.tanh(sim_query_embedding)

        #cos_similarity余弦相似度[batch_size, similarity]
        query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_embedding), axis=-1), name='query_norm')
        sim_query_norm = tf.sqrt(tf.reduce_sum(tf.square(sim_query_embedding), axis=-1), name='sim_query_norm')

        dot = tf.reduce_sum(tf.multiply(query_embedding, sim_query_embedding), axis=-1)
        cos_similarity = tf.divide(dot, (query_norm*sim_query_norm), name='cos_similarity')
        self.similarity = cos_similarity

        #对于只有一个负样本的情况处理loss,根据阈值确定正负样本
        # 预测为正例的概率
        cond = (self.similarity > self.config["neg_threshold"])

        zeros = tf.zeros_like(self.similarity, dtype=tf.float32)
        ones = tf.ones_like(self.similarity, dtype=tf.float32)
        # pred_neg_prob = tf.where(cond, tf.square(self.similarity), zeros)
        # logits = [[pred_pos_prob[i], pred_neg_prob[i]] for i in range(self.batch_size)]

        # cond = (self.similarity > 0.0)
        pos = tf.where(cond, tf.square(self.similarity), 1 - tf.square(self.similarity))
        neg = tf.where(cond, 1 - tf.square(self.similarity), tf.square(self.similarity))
        logits = [[neg[i], pos[i]] for i in range(self.config['batch_size'])]

        self.logits = logits
        outputs = dict(logits=self.logits)

        super(DSSM, self).__init__(inputs=[query, sim_query], outputs=outputs)



