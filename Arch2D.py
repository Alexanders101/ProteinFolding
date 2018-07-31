import numpy as np
import tensorflow as tf

from tensorflow import keras
from ProteinNetworkUtils import Lattice2D

def make_short_network_2D(max_aa):
    inp = keras.layers.Input(shape=(4, max_aa), dtype=tf.int64)

    # Reshape Inputs
    acids = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    mask = keras.layers.Lambda(lambda x: tf.cast(x[:, 1], tf.bool))(inp)
    aa_length = keras.layers.Lambda(lambda x: tf.count_nonzero(x, axis=1, dtype=tf.int64))(mask)
    indices = keras.layers.Lambda(lambda x: tf.transpose(x[:, 2:4, :], perm=(0, 2, 1)))(inp)

    lattice = Lattice2D(max_aa)([acids, mask, indices])
    conv = keras.layers.Conv2D(16, (5,5))(lattice)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(32, (3,3))(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(64, (3,3))(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(128, (3,3))(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D(2)(conv)
    conv = keras.layers.Conv2D(128, (2,2))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.Flatten()(conv)
    
    acids = keras.layers.Reshape((max_aa, 1))(acids)
    reverse = keras.layers.Lambda(lambda x: tf.reverse_sequence(x[0], seq_lengths=tf.cast(x[1], tf.int64), seq_axis=1))([acids, aa_length])
    
    forward_lstm = keras.layers.CuDNNGRU(64, return_sequences=False)(acids)
    backward_lstm = keras.layers.CuDNNGRU(64, return_sequences=False)(reverse)
    bi_lstm = keras.layers.concatenate([forward_lstm, backward_lstm], axis=1)

    combined = keras.layers.concatenate([conv, bi_lstm], axis=1)
    
    policy = keras.layers.Dense(128, activation='relu')(combined)
    policy = keras.layers.Dense(4, activation=None)(policy)

    value = keras.layers.Dense(128, activation='relu')(combined)
    value = keras.layers.Dense(1, activation=None)(value)

    model = keras.Model(inp, [policy, value])
    return model

def make_short_network_2D_greedy(max_aa):
    inp = keras.layers.Input(shape=(4, max_aa), dtype=tf.int64)

    # Reshape Inputs
    acids = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    mask = keras.layers.Lambda(lambda x: tf.cast(x[:, 1], tf.bool))(inp)
    aa_length = keras.layers.Lambda(lambda x: tf.count_nonzero(x, axis=1, dtype=tf.int64))(mask)
    indices = keras.layers.Lambda(lambda x: tf.transpose(x[:, 2:4, :], perm=(0, 2, 1)))(inp)

    lattice = Lattice2D(max_aa)([acids, mask, indices])
    conv = keras.layers.Conv2D(16, (5,5))(lattice)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(32, (3,3))(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(64, (3,3))(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D()(conv)
    conv = keras.layers.Conv2D(128, (3,3))(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.MaxPooling2D(2)(conv)
    conv = keras.layers.Conv2D(128, (2,2))(conv)
    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation('relu')(conv)
    conv = keras.layers.Flatten()(conv)
    
    acids = keras.layers.Reshape((max_aa, 1))(acids)
    reverse = keras.layers.Lambda(lambda x: tf.reverse_sequence(x[0], seq_lengths=tf.cast(x[1], tf.int64), seq_axis=1))([acids, aa_length])
    
    forward_lstm = keras.layers.GRU(64, return_sequences=False)(acids)
    backward_lstm = keras.layers.GRU(64, return_sequences=False)(reverse)
    bi_lstm = keras.layers.concatenate([forward_lstm, backward_lstm], axis=1)

    combined = keras.layers.concatenate([conv, bi_lstm], axis=1)
    
    policy = keras.layers.Dense(128, activation='relu')(combined)
    policy = keras.layers.Dense(4, activation='softmax')(policy)

    value = keras.layers.Dense(128, activation='relu')(combined)
    value = keras.layers.Dense(1, activation=None)(value)

    model = keras.Model(inp, [policy, value])
    return model
