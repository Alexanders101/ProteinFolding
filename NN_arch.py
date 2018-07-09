import numpy as np
import tensorflow as tf
import keras
from Data_prep import read_data, partial_format
from ProteinNetworkUtils import Lattice, RemoveMask

#filename = "/Users/danielstephens/PycharmProjects/Proteinfolding/venv/small_data_partial.h5"
#save, Y, sizes = read_data(filename)


def make_network(max_aa):
    max_lattice = 2*max_aa -1


    inp = keras.layers.Input(shape=(5, max_aa), dtype=tf.int64)
    inp1 = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    inp2 = keras.layers.Lambda(lambda x: tf.cast(x[:, 1], tf.bool))(inp)
    inp3_temp = keras.layers.Lambda(lambda x: tf.transpose(x[:, 2:5, :], perm=(0, 2, 1)))(inp)

    inp3 = Lattice(max_aa)([inp1, inp3_temp, inp2])

    inp1 = keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(inp1)
    inp2 = keras.layers.Lambda(lambda x: tf.expand_dims(x, -1))(inp2)
    inp3 = keras.layers.Lambda(lambda x: tf.expand_dims(x, -1), output_shape=(max_lattice, max_lattice, max_lattice, 1))(inp3)

    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(5, return_sequences=True))(inp1)

    masked = keras.layers.Lambda(lambda x: x[0] * tf.cast(x[1], tf.float32))([inp1, inp2])
    masked = keras.layers.Masking()(masked)
    lstm = keras.layers.LSTM(5, return_sequences=True)(masked)

    lstm_comb = keras.layers.concatenate([bi_lstm, lstm], axis=2)
    #lstm_comb = keras.layers.Flatten()(lstm_comb)
    lstm_comb_unmask = RemoveMask()(lstm_comb)
    lstm_comb = keras.layers.LSTM(32)(lstm_comb)
    lstm_comb = keras.layers.Reshape((1, 32))(lstm_comb)
    final = keras.layers.LSTM(32, return_sequences=True)(lstm_comb_unmask)
    final = keras.layers.Lambda(lambda x: x[:,-5:,:])(final)

    final_comb = keras.layers.concatenate([lstm_comb, final], axis=1)
    final_actually = keras.layers.Flatten()(final_comb)


    mid = keras.layers.Conv3D(16, (3, 3, 3))(inp3)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.MaxPooling3D(strides=3)(mid)
    mid = keras.layers.Activation('relu')(mid)

    mid = keras.layers.Conv3D(32, (3, 3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.MaxPooling3D(strides=3)(mid)
    mid = keras.layers.Activation('relu')(mid)

    mid = keras.layers.Conv3D(64, (3, 3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.MaxPooling3D(strides=3)(mid)
    mid = keras.layers.Activation('relu')(mid)

    mid = keras.layers.Conv3D(128, (3, 3, 3))(mid)
    mid = keras.layers.BatchNormalization()(mid)
    mid = keras.layers.Activation('relu')(mid)
    mid = keras.layers.Flatten()(mid)

    total = keras.layers.concatenate([mid, final_actually], axis=1)

    total = keras.layers.Dense(128, activation="relu")(total)
    total = keras.layers.Dense(1, activation=None)(total)

    model = keras.Model(inp, total)
    model.compile("adam", loss="mse")
    return model