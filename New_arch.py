import numpy as np
import tensorflow as tf
import keras
from Data_prep import read_data, partial_format
from ProteinNetworkUtils import Lattice, RemoveMask


def make_network2(max_aa):
    inp = keras.layers.Input(shape=(max_aa, 5, 5, 5, 1), dtype=tf.float64)
    inp1 = keras.layers.Input(shape=(max_aa, 1), dtype=tf.float32)
    inp2 = keras.layers.Input(shape=(max_aa, 1), dtype=tf.int64)
    inp2 = keras.layers.Lambda(lambda x: tf.cast(x, tf.bool))(inp2)

    # added these three lines
    bool_mask = keras.layers.Lambda(lambda x: tf.cast(x, tf.bool))(inp1)
    aa_length = keras.layers.Lambda(lambda x: tf.count_nonzero(x, axis=1))(bool_mask)

    inp = keras.layers.TimeDistributed(keras.layers.Conv3D(32, (3, 3, 3), padding="valid"))(inp)
    inp = keras.layers.TimeDistributed(keras.layers.Conv3D(128, (3, 3, 3), padding="valid"))(inp)
    inp = keras.layers.Reshape((50, 128), input_shape=(50, 1, 1, 1, 128))(inp)

    reversed = keras.layers.Lambda(lambda x: tf.reverse_sequence(x[0], x[1], seq_axis=0))([inp, aa_length])

    # switch bidirection with two lstm marked later on.

    # forward_lstm and backward_lstm are the two

    forward_lstm = keras.layers.LSTM(10, return_sequences=True)(inp)
    backward_lstm = keras.layers.LSTM(10, return_sequences=True)(reversed)
    bi_lstm = keras.layers.concatenate([forward_lstm, backward_lstm], axis=2)

    masked = keras.layers.Lambda(lambda x: x[0] * tf.cast(x[1], tf.float32))([inp, inp2])
    masked = keras.layers.Masking()(masked)
    lstm = keras.layers.LSTM(10, return_sequences=True)(masked)

    lstm_comb = keras.layers.concatenate([bi_lstm, lstm], axis=2)
    lstm_comb_unmask = RemoveMask()(lstm_comb)
    lstm_comb = keras.layers.LSTM(32)(lstm_comb)

    lstm_comb = keras.layers.Reshape((1, 32))(lstm_comb)
    final = keras.layers.LSTM(32, return_sequences=True)(lstm_comb_unmask)
    final = keras.layers.Lambda(lambda x: x[:, -5:, :])(final)

    final_comb = keras.layers.concatenate([lstm_comb, final], axis=1)
    final_actually = keras.layers.Flatten()(final_comb)
    total = keras.layers.Dense(128, activation="relu")(final_actually)
    total = keras.layers.Dense(1, activation=None)(total)

    model = keras.Model(inp, total)
    model.compile("adam", loss="mse")
    return model