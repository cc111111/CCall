# -*- coding: utf-8 -*-
# @Time    : 2020/2/14 20:27
# @Author  : Dai PuWei
# @Email   : 771830171@qq.com
# @File    : MNIST2MNIST_M.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input, Embedding, Lambda
import tensorflow.keras.backend as K

K.clear_session()

def build_w2vm(word_size, window, nb_word, nb_negative):
    K.clear_session()  # 清除之前的模型，省得压满内存
    # CBOW输入
    input_words = Input(shape=(window * 2,), dtype='int32')
    input_vecs = Embedding(nb_word, word_size, name='word2vec')(input_words)
    input_vecs_sum = Lambda(lambda x: K.sum(x, axis=1))(input_vecs)  # CBOW模型，直接将上下文词向量求和

    # 构造随机负样本，与目标组成抽样
    target_word = Input(shape=(1,), dtype='int32')
    negatives = Lambda(lambda x: K.random_uniform((K.shape(x)[0], nb_negative), 0, nb_word, 'int32'))(target_word)
    samples = Lambda(lambda x: K.concatenate(x))([target_word, negatives])  # 构造抽样，负样本随机抽。负样本也可能抽到正样本，但概率小。

    # 只在抽样内做Dense和softmax
    softmax_weights = Embedding(nb_word, word_size, name='W')(samples)
    softmax_biases = Embedding(nb_word, 1, name='b')(samples)
    softmax = Lambda(lambda x:
                     K.softmax((K.batch_dot(x[0], K.expand_dims(x[1], 2)) + x[2])[:, :, 0])
                     )([softmax_weights, input_vecs_sum, softmax_biases])  # 用Embedding层存参数，用K后端实现矩阵乘法，以此复现Dense层的功能

    # 留意到，我们构造抽样时，把目标放在了第一位，也就是说，softmax的目标id总是0，这可以从data_generator中的z变量的写法可以看出

    model = Model(inputs=[input_words, target_word], outputs=softmax)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 请留意用的是sparse_categorical_crossentropy而不是categorical_crossentropy
    model.summary()
    return model



# nb_word = 256
    # word_size = 256
    # input_words = Input(shape=(64 * 2,), dtype='int32')  # window=64
    # model = tf.keras.Sequential([Embedding(nb_word, word_size, name='word2vec')(input_words)
    #
    #
    # ])
def build_feature_extractor():
    """
    这是特征提取子网络的构建函数
    :param image_input: 图像输入张量
    :param name: 输出特征名称
    """
    model = tf.keras.Sequential([Embedding(256, 100, name='word2vec')
                                 # Conv2D(filters=32, kernel_size=5,strides=1),
                                 # #tf.keras.layers.BatchNormalization(),
                                 # Activation('relu'),
                                 # MaxPool2D(pool_size=(2, 2), strides=2),
                                 # Conv2D(filters=48, kernel_size=5,strides=1),
                                 # #tf.keras.layers.BatchNormalization(),
                                 # Activation('relu'),
                                 # MaxPool2D(pool_size=(2, 2), strides=2),
                                 # Flatten(),
    ])
    return model

def build_image_classify_extractor():
    """
    这是搭建图像分类器模型的函数
    :param image_classify_feature: 图像分类特征张量
    :return:
    """
    model = tf.keras.Sequential([Dense(100),
                                 #tf.keras.layers.BatchNormalization(),
                                 Activation('relu'),
                                 #tf.keras.layers.Dropout(0.5),
                                 Dense(100,activation='relu'),
                                 #tf.keras.layers.Dropout(0.5),
                                 Dense(2,activation='softmax',name="image_cls_pred"),
    ])
    return model

def build_domain_classify_extractor():
    """
    这是搭建域分类器的函数
    :param domain_classify_feature: 域分类特征张量
    :return:
    """
    # 搭建域分类器
    model = tf.keras.Sequential([Dense(100),
                                 #tf.keras.layers.BatchNormalization(),
                                 Activation('relu'),
                                 #tf.keras.layers.Dropout(0.5),
                                 Dense(2, activation='softmax', name="domain_cls_pred")
    ])
    return model
