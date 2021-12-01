import tensorflow as tf

class Siamese(tf.keras.Model):
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
        # 利用词嵌入矩阵将输入数据转成词向量，shape=[batch_size, seq_len, embedding_size]
        class GatherLayer(tf.keras.layers.Layer):
            def __init__(self, config, vocab_size, word_vectors):
                super(GatherLayer, self).__init__()
                self.config = config

                self.vocab_size = vocab_size
                self.word_vectors = word_vectors

            def build(self, input_shape):
                with tf.name_scope('embedding'):
                    if not self.config['use_word2vec']:
                        self.embedding_w = tf.Variable(tf.keras.initializers.glorot_normal()(
                            shape=[self.vocab_size, self.config['embedding_size']],
                            dtype=tf.float32), trainable=True, name='embedding_w')
                    else:
                        self.embedding_w = tf.Variable(tf.cast(self.word_vectors, tf.float32), trainable=True,
                                                       name='embedding_w')
                self.build = True

            def call(self, inputs, **kwargs):
                return tf.gather(self.embedding_w, inputs, name='embedded_words')

            def get_config(self):
                config = super(GatherLayer, self).get_config()

                return config



        # class shared_net(tf.keras.Model):
        #     def __init__(self, config, vocab_size, word_vectors):
        #         query = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, name='input_x_ids')
        #         query_embedding = GatherLayer(config, vocab_size, word_vectors)(query)
        #         query_embedding_output = shared_lstm_layer(config)(query_embedding)
        #
        #         super(shared_net, self).__init__(inputs=[query], outputs=query_embedding_output)
        shared_net = tf.keras.Sequential([GatherLayer(config, vocab_size, word_vectors),
                                          shared_lstm_layer(config)])

        query_embedding_output = shared_net.predict_step(query)
        sim_query_embedding_output = shared_net.predict_step(sim_query)

        # #bi-lstm孪生网络
        # forward_layer_1 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
        #                                      return_sequences=True)
        # backward_layer_1 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
        #                                       return_sequences=True, go_backwards=True)
        # forward_layer_2 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
        #                                        return_sequences=True)
        # backward_layer_2 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
        #                                         return_sequences=True, go_backwards=True)
        # # 双向网络层，得到的结果拼接，维度大小为[batch_size, forward_size+backward_size]
        # query_res_1 = tf.keras.layers.Bidirectional(forward_layer_1, backward_layer=backward_layer_1)(query_embedding)
        # sim_query_res_1 = tf.keras.layers.Bidirectional(forward_layer_1, backward_layer=backward_layer_1)(sim_query_embedding)
        #
        # #层间dropout
        # query_res_1 = tf.keras.layers.Dropout(0.4)(query_res_1)
        # sim_query_res_1 = tf.keras.layers.Dropout(0.4)(sim_query_res_1)
        #
        # #第二层bi-lstm
        # query_res_2 = tf.keras.layers.Bidirectional(forward_layer_2, backward_layer=backward_layer_2)(query_res_1)
        # sim_query_res_2 = tf.keras.layers.Bidirectional(forward_layer_2, backward_layer=backward_layer_2)(sim_query_res_1)
        #
        # #摊平
        # tmp_query_embedding = tf.reshape(query_res_2, [self.config['batch_size'],
        #                                                self.config['seq_len'] * self.config['hidden_size']])
        #
        # tmp_sim_query_embedding = tf.reshape(sim_query_res_2, [self.config['batch_size'],
        #                                                        self.config['seq_len'] * self.config['hidden_size']])
        #
        # #全连接层
        # query_embedding_output = tf.keras.layers.Dense(self.config['output_size'])(tmp_query_embedding)
        # sim_query_embedding_output = tf.keras.layers.Dense(self.config['output_size'])(tmp_sim_query_embedding)

        #余弦函数计算相似度
        # cos_similarity余弦相似度[batch_size, similarity]
        query_norm = tf.sqrt(tf.reduce_sum(tf.square(query_embedding_output), axis=-1), name='query_norm')
        sim_query_norm = tf.sqrt(tf.reduce_sum(tf.square(sim_query_embedding_output), axis=-1), name='sim_query_norm')

        dot = tf.reduce_sum(tf.multiply(query_embedding_output, sim_query_embedding_output), axis=-1)
        cos_similarity = tf.divide(dot, (query_norm * sim_query_norm), name='cos_similarity')
        self.similarity = cos_similarity

        # 预测为正例的概率
        cond = (self.similarity > self.config["neg_threshold"])
        pos = tf.where(cond, tf.square(self.similarity), 1 - tf.square(self.similarity))
        neg = tf.where(cond, 1 - tf.square(self.similarity), tf.square(self.similarity))
        predictions = [[neg[i], pos[i]] for i in range(self.config['batch_size'])]

        self.logits = self.similarity
        outputs = dict(logits=self.logits, predictions=predictions)

        super(Siamese, self).__init__(inputs=[query, sim_query], outputs=outputs)


class shared_lstm_layer(tf.keras.layers.Layer):
    '''
    共享lstm层参数
    '''
    def __init__(self, config):
        self.config = config
        super(shared_lstm_layer, self).__init__()

    def build(self, input_shape):
        forward_layer_1 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                               return_sequences=True)
        backward_layer_1 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                                return_sequences=True, go_backwards=True)
        forward_layer_2 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                               return_sequences=True)
        backward_layer_2 = tf.keras.layers.LSTM(self.config['hidden_size'], dropout=self.config['dropout_rate'],
                                                return_sequences=True, go_backwards=True)
        self.bilstm_1 = tf.keras.layers.Bidirectional(forward_layer_1, backward_layer=backward_layer_1)
        self.bilstm_2 = tf.keras.layers.Bidirectional(forward_layer_2, backward_layer=backward_layer_2)
        self.layer_dropout = tf.keras.layers.Dropout(0.4)
        self.output_dense = tf.keras.layers.Dense(self.config['output_size'])

        super(shared_lstm_layer, self).build(input_shape)

    def get_config(self):
        config = {}
        return config

    def call(self, inputs, **kwargs):
        query_res_1 = self.bilstm_1(inputs)
        query_res_1 = self.layer_dropout(query_res_1)
        query_res_2 = self.bilstm_2(query_res_1)

        #取时间步的平均值，摊平[batch_size, forward_size+backward_size]
        avg_query_embedding = tf.reduce_mean(query_res_2, axis=1)
        tmp_query_embedding = tf.reshape(avg_query_embedding, [self.config['batch_size'], self.config['hidden_size']*2])
        # 全连接层[batch_size, dense_dim]
        query_embedding_output = self.output_dense(tmp_query_embedding)
        query_embedding_output = tf.keras.activations.relu(query_embedding_output)
        return query_embedding_output







