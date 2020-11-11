from keras.models import Model
from keras.layers import Input, Conv1D, Dropout, MaxPooling1D,LSTM
from keras.layers.core import Dense, Activation

def bn_relu(layer):
    from keras.layers import BatchNormalization
    from keras.layers import Activation
    layer = BatchNormalization()(layer)
    layer = Activation('relu')(layer)
    return layer

def conv_layers(layer, filters=16, kernel_size=3, strides=1, is_train=True):
    layer = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        kernel_initializer='he_normal',
        trainable=is_train) (layer)

    layer = bn_relu(layer)
    return layer

def compile(model):
    from keras.optimizers import Adam
    optimizer = Adam(lr=0.001, clipnorm=1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def add_cnn(inputs):
    # 1
    layer = conv_layers(inputs, filters=16, kernel_size=15, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 2
    layer = conv_layers(layer, filters=16, kernel_size=15, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 3
    layer = conv_layers(layer, filters=32, kernel_size=11, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 4
    layer = conv_layers(layer, filters=32, kernel_size=11, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 5
    layer = conv_layers(layer, filters=64, kernel_size=7, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 6
    layer = conv_layers(layer, filters=64, kernel_size=7, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 7
    layer = conv_layers(layer, filters=128, kernel_size=3, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    # 8
    layer = conv_layers(layer, filters=128, kernel_size=3, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=2)(layer)
    layer = Dropout(0.2, seed=1)(layer)
    return layer

def build_network(**params):
    input = Input(shape=params['input_shape'],
                dtype='float32',
                name='ecg_inputs')
    cnn_output = add_cnn(input)
    lstm1 = LSTM(units=64,return_sequences=True)(cnn_output)
    lstm2 = LSTM(units=64)(lstm1)
    fc = Dense(params['num_categories'])(lstm2)
    output = Activation('softmax')(fc)
    model = Model(inputs=input, outputs=output)
    compile(model)
    return model