import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Dense, Dropout, Flatten


def cnn_backbones(class_num, input_shape, convs,  filter):
    inputs = Input(shape=input_shape)

    # 第一个卷积块
    for i in range(convs):
        x = Conv1D(filters=filter, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)

    # 为了实现残差连接，我们加入一个跳跃连接，连接第一个卷积块的输出到第二个卷积块的输出
    y = Add()([x, inputs])

    # 第二个卷积块
    x = Conv1D(filters=filter, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(filters=filter, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    z = Add()([y, x])

    x = Flatten()(x)

    # 全连接层和输出层
    outputs = Dense(class_num, activation='softmax')(x)
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs)
    return model


def cnn_with_embedding(class_num, input_shape, convs, embed_dimension, filter):
    # 因为embedding的输出已经被squeeze了最后一个维度，所以我们直接使用这个shape
    # 作为cnn模型的输入shape（这里的32是embedding的输出维度）
    cnn_model = cnn_backbones(class_num, (input_shape, embed_dimension), convs , filter)
    # 将模型合并
    inputs = Input(shape=(input_shape, 1))
    x = Embedding(input_dim=3030, output_dim=embed_dimension)(inputs)
    x = tf.squeeze(x, axis=-2)
    outputs = cnn_model(x)
    # 创建大模型
    combined_model = Model(inputs=inputs, outputs=outputs)
    return combined_model