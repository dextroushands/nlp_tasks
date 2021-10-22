import tensorflow as tf

from models.model_base import BaseModel


class TextCnn(BaseModel):
    '''
    Textcnn模型
    '''
    def __init__(self, inputs, model_config, word_vectors, vocab_size):
        self.config = model_config
        self.embedding = word_vectors
        self.vocab_size = vocab_size
        self.inputs = inputs['x']
        self.labels = inputs['y']
        self.l2_loss = 0

        word_ids = tf.keras.layers.Input(
            shape=(None, ), dtype=tf.int32, name='input_word_ids'
        )

        #模型网络结构
        with tf.name_scope('embedding'):
            if self.embedding == None:
                embedding_w = tf.Variable(tf.keras.initializers.glorot_normal()(shape=[self.vocab_size, self.config['emebdding_size']],
                                                                    detype=tf.float32), name='embedding_w')
            else:
                embedding_w = tf.Variable(tf.cast(self.embedding, tf.float32), name='embedding_w')

            #利用词嵌入矩阵将输入数据转成词向量，shape=[batch_size, seq_len, embedding_size]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs, name='embedded_words')
            #卷积输入是四维向量，需要增加维度[batch_size, height, width, channels]
            expanded_words = tf.expand_dims(embedded_words, axis=-1, name='expanded_words')

        #进行卷积和池化操作
        pooled_outputs = []
        #创建几种不同尺寸的卷积核提取文本信息，通常使用长度为3，4，5等长度
        for filter in self.config['conv_filters']:
            #初始化卷积核权重和偏置
            #[filter_height, filter_width, in_channels, out_channels],先矩阵相乘然后求和
            filter_size = [filter, self.config['embedding_size'], 1, self.config['num_filters']]
            conv_w = tf.keras.initializers.truncated_normal(filter_size, stddev=0.1)
            conv_b = tf.constant(0.1, shape=self.config['num_filters'])
            conv = tf.nn.conv2d(
                expanded_words,
                conv_w,
                strides=[1,1,1,1],
                padding='VALID',
                name='conv'
            )
            #relu函数的非线性映射,得到卷积的结果[batch_size, seq_len-filter+1, 1, num_filters]
            h = tf.nn.relu(tf.nn.bias_add(conv, conv_b))
            #池化层，最大池化
            pooled = tf.nn.max_pool(
                h,
                #ksize,shape[batch_size, height, width, channels]
                ksize=[1, self.config['seq_len']-filter_size+1, 1, 1],
                strides=[1,1,1,1],
                padding='VALID',
                name='max_pooling'
            )
            #每种filter得到num_filters个池化结果
            pooled_outputs.append(pooled)

        #最终输出的cnn长度
        total_cnn_output = self.config['num_filters'] * len(self.config['conv_filters'])

        #将不同的卷积核根据channel concat起来
        h_pool = tf.concat(pooled_outputs, 3)

        #摊平成二维输入到全连接层[batch_size, total_cnn_output]
        h_pool_flat = tf.reshape(h_pool, [-1, total_cnn_output])

        #dropout层
        h_drop_out = tf.nn.dropout(h_pool_flat, self.config['keep_prob'])

        #全连接层的输出
        with tf.name_scope('output'):
            output_w = tf.keras.initializers.glorot_normal()(shape=[total_cnn_output, self.config['num_classes']])
            output_b = tf.constant(0.1, shape=[self.config['num_classes']])
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)

            self.logits = tf.matmul(h_drop_out, output_w) + output_b
            self.loss = self.cal_loss() + self.config['l2_reg_lambda'] * self.l2_loss
            self.predictions = self.get_predictions(self.logits)

        super(TextCnn, self).__init__(inputs=[self.word_ids], outputs=self.logits, **kwargs)

    def cal_loss(self):
        '''
        计算分类问题的损失
        :param logits:
        :return:
        '''
        loss = 0.0
        if self.config['num_classes']>1:
            self.labels = tf.cast(self.labels, dtype=tf.int32)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.labels)
        elif self.config['num_classes'] == 1:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.reshape(self.labels, [-1, 1]))

        loss = tf.reduce_mean(loss)
        return loss














