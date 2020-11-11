from keras.models import Model
from keras.layers import Input, Conv1D,  Dropout, MaxPooling1D, Concatenate,LSTM
from keras.layers.core import Dense, Activation
from keras import backend as K

def recall(y_true, y_pred):
    # Calculates the recall
    y_true = y_true[:, 1]
    y_pred = y_pred[:, 1]
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    positive = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positive / (positive + K.epsilon())

def spe(y_true, y_pred):
    # Calculates the recall
    y_true = y_true[:, 0]
    y_pred = y_pred[:, 0]
    negative = K.sum(K.round(K.clip(y_true, 0, 1)))
    true_negative = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    return true_negative / (negative + K.epsilon())

def F1(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    # if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
    #    return 0
    p = spe(y_true, y_pred)
    r = recall(y_true, y_pred)
    fbeta_score = 2 * (p * r) / (p + r + K.epsilon())
    return fbeta_score


def bnrelu(layer):
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
        trainable=is_train,
    )(layer)
    layer = bnrelu(layer)
    return layer

def compile(model):
    from keras.optimizers import Adam
    optimizer = Adam(
        lr=0.001,
        clipnorm=1)
    # categorical_crossentropy
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

def build_pcg_network2(inputs):
    # 1
    layer = conv_layers(inputs, filters=16, kernel_size=5, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=5)(layer)
    layer = Dropout(0.2, seed=1)(layer)

    # 2
    layer = conv_layers(layer, filters=16, kernel_size=5, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=5)(layer)
    layer = Dropout(0.2, seed=1)(layer)

    # 3
    layer = conv_layers(layer, filters=16, kernel_size=5, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=5)(layer)
    layer = Dropout(0.2, seed=1)(layer)

    # 4
    layer = conv_layers(layer, filters=32, kernel_size=5, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=4)(layer)
    layer = Dropout(0.2, seed=1)(layer)

    # 5
    layer = conv_layers(layer, filters=32, kernel_size=5, strides=1, is_train=True)
    layer = MaxPooling1D(pool_size=4)(layer)
    layer = Dropout(0.2, seed=1)(layer)

    return layer


def build_network(**params):
    inputs_pcg = []
    for i in range(4):
        inputs_pcg.append(Input(shape=params['input_shape'],
                                dtype='float32',
                                name='pcg_inputs' + str(i)))
    cnn_output = []
    for i in range(4):
        cnn_output.append(build_pcg_network2(inputs_pcg[i]))

    layer = Concatenate(axis=-1)(cnn_output)

    layer = LSTM(units=128,return_sequences=True)(layer)
    layer = LSTM(units=64,return_sequences=False)(layer)

    layer = Dense(2)(layer)
    output = Activation('softmax')(layer)

    model = Model(inputs=inputs_pcg, outputs=output)

    compile(model)
    return model
