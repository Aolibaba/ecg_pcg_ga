# coding=utf-8
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, concatenate, Dropout, AveragePooling1D, \
    GlobalAveragePooling1D, Average ,Add,Lambda
from keras.layers.core import Dense, Activation
from keras.layers.wrappers import TimeDistributed
from keras.utils import conv_utils
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import math
from keras.layers.core import Layer
from keras.engine import InputSpec
from keras import backend as K
from keras.regularizers import l2, l1, l1_l2


class MyLayer(Layer):

    def __init__(self, **kwargs):
        self.input_shape1=[1, 1, 1]
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec=[InputSpec(shape=input_shape)]

    def compute_output_shape(self, input_shape):
        output_shape=(input_shape[0], 1, input_shape[2])
        return output_shape

    def call(self, inputs, mask=None):
        # out_put =K.sum(inputs,axis=1,keepdims=True)
        out_put=tf.reduce_mean(inputs, axis=1, keepdims=True)
        return out_put  # remove dummy last dimension


def _bn_relu(layer):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer=BatchNormalization()(layer)
    layer=Activation('relu')(layer)

    return layer


def add_conv_layers(layer, filters=16, kernel_size=3, strides=1, is_train=True):
    layer=Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        trainable=is_train,
        ) \
        (layer)
    # W_regularizer=l2(0.005)
    layer=_bn_relu(layer)
    return layer


def add_compile(model):
    from keras.optimizers import Adam
    optimizer=Adam(
        lr=0.01,
        clipnorm=1)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['accuracy']
                  )
def build_ecg_network(inputs, **params):
    # 1
    layer=add_conv_layers(inputs, filters=32, kernel_size=16, strides=1, is_train=True)
    layer=AveragePooling1D(pool_size=4)(layer)
    layer=Dropout(0.2, seed=1)(layer)
    # 2
    layer=add_conv_layers(layer, filters=32, kernel_size=16, strides=1, is_train=True)
    #layer=Dropout(0.2,seed=1)(layer)
    layer=AveragePooling1D(pool_size=4)(layer)
    # 3
    layer=add_conv_layers(layer, filters=64, kernel_size=16, strides=1, is_train=True)
    layer=AveragePooling1D(pool_size=4)(layer)
    layer=Dropout(0.2, seed=1)(layer)
    # 4
    layer=add_conv_layers(layer, filters=64, kernel_size=16, strides=1, is_train=True)
    #layer=Dropout(0.2,seed=1)(layer)
    layer=AveragePooling1D(pool_size=4)(layer)
    return layer

def build_network(**params):
    inputs_ecg=Input(shape=params['input_shape'],
                     dtype='float32',
                     name='ecg_inputs')
    inputs_pcg = Input(shape=[72], dtype='float32', name='pcg_inputs')
    ecg_net = build_ecg_network(inputs_ecg, **params)
    ecg_net = GlobalAveragePooling1D()(ecg_net)

    layer = concatenate([ecg_net,inputs_pcg])
    layer = Dense(64, name='layer_mul1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(32, name='layer_mul2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(2, name='layer_mul3')(layer)
    out_put = Activation('softmax', name='outputs')(layer)

    model = Model(inputs=[inputs_ecg,inputs_pcg], outputs=[out_put])
    add_compile(model)
    return model

def build_network2(**params):
    inputs_ecg=Input(shape=params['input_shape'],
                     dtype='float32',
                     name='ecg_inputs')
    ecg_net = build_ecg_network(inputs_ecg, **params)
    ecg_net = GlobalAveragePooling1D()(ecg_net)
    layer = Dense(64, name='layer_mul1')(ecg_net)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(32, name='layer_mul2')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(2, name='layer_mul3')(layer)
    out_put_ecg = Activation('softmax',name = 'outputs')(layer)

    inputs_pcg=Input(shape=[318], dtype='float32', name='pcg_inputs')
    layer = Dense(318,W_regularizer=l1(0.01))(inputs_pcg)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(250,W_regularizer=l1(0.01))(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(125,W_regularizer=l1(0.01))(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(25,W_regularizer=l1(0.01))(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.2,seed=1)(layer)
    layer = Dense(2)(layer)
    out_put_pcg = Activation('softmax',name='outputs')(layer)

    weight_1 = Lambda(lambda x: x * 1)
    weight_2 = Lambda(lambda x: x * 0)
    weight_gru1 = weight_1(out_put_ecg)
    weight_gru2 = weight_2(out_put_pcg)
    #out_put = Add(name = 'outputs')([weight_gru1, weight_gru2])

    model=Model(inputs=[inputs_ecg], outputs=[out_put_ecg])
    #model = Model(inputs=[inputs_ecg,inputs_pcg], outputs=[out_put_pcg])
    add_compile(model)
    return model
if __name__ == '__main__':
    from keras.models import load_model
    print('over')