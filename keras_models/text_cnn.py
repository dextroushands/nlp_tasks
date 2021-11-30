import tensorflow as tf


class TextCnn(tf.keras.Model):
    '''
    构建textcnn网络结构
    '''
    def __init__(self, config, vocab_size, word_vectors):
        self.config = config
        self.vocab_size = vocab_size
        self.word_vectors = word_vectors

        #输入层
        word_ids = tf.keras.layers.Input(shape=(None, ), dtype=tf.int64, name='input_word_ids')

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

        #卷积输入是四维向量，需要增加维度[batch_size, height, width, channels]
        expanded_words = tf.expand_dims(embedded_words, axis=-1, name='expanded_words')

        # 进行卷积和池化操作
        pooled_outputs = []
        # 创建几种不同尺寸的卷积核提取文本信息，通常使用长度为3，4，5等长度
        for filter in self.config['conv_filters']:
            # 初始化卷积核权重和偏置
            # [filter_height, filter_width, in_channels, out_channels],先矩阵相乘然后求和
            # filter_size = [filter, self.config['embedding_size'], 1, self.config['num_filters']]
            #kenel_size [height, width]
            kernel_size = [filter, self.config['embedding_size']]
            #output_size
            output_size = self.config['num_filters']
            # conv_w = tf.keras.initializers.truncated_normal(filter_size, stddev=0.1)
            # conv_b = tf.constant(0.1, shape=self.config['num_filters'])
            h = tf.keras.layers.Conv2D(
                filters=output_size,
                kernel_size=kernel_size,
                strides=(1,1),
                kernel_initializer=tf.keras.initializers.truncated_normal(),
                bias_initializer=tf.keras.initializers.constant(0.1),
                activation='relu',
                padding='VALID',
                data_format="channels_last",
                input_shape=(self.config['seq_len'], self.config['embedding_size'], 1)
            )(expanded_words)
            # relu函数的非线性映射,得到卷积的结果[batch_size, seq_len-filter+1, 1, num_filters]
            # h = tf.keras.layers.Activation.relu(tf.keras.layers.Layer.bias_add(conv, conv_b))
            # 池化层，最大池化
            pooled = tf.keras.layers.MaxPooling2D(
                # ksize,shape[batch_size, height, width, channels]
                pool_size=[self.config['seq_len'] - filter+ 1, 1],
                strides=(1,1),
                padding='VALID',
                data_format="channels_last"
            )(h)
            # 每种filter得到num_filters个池化结果
            pooled_outputs.append(pooled)


        # 最终输出的cnn长度
        total_cnn_output = self.config['num_filters'] * len(self.config['conv_filters'])

        # 将不同的卷积核根据channel concat起来
        h_pool = tf.concat(pooled_outputs, 3)

        # 摊平成二维输入到全连接层[batch_size, total_cnn_output]
        h_pool_flat = tf.reshape(h_pool, [-1, total_cnn_output])

        # dropout层
        h_drop_out = tf.keras.layers.Dropout(self.config['dropout_rate'])(h_pool_flat)

        # 全连接层的输出
        with tf.name_scope('output'):
            output_w = tf.keras.initializers.glorot_normal()(shape=[total_cnn_output, self.config['num_classes']])
            output_b = tf.constant(0.1, shape=[self.config['num_classes']])
            # self.l2_loss += tf.nn.l2_loss(output_w)
            # self.l2_loss += tf.nn.l2_loss(output_b)

            # self.logits = tf.matmul(h_drop_out, output_w) + output_b
            self.logits = tf.keras.layers.Dense(self.config["num_classes"])(h_drop_out)
        #输出为[batch_size, num_classes]
        outputs = dict(logits=self.logits)


        super(TextCnn, self).__init__(inputs=[word_ids], outputs=outputs)

