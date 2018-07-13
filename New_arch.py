import numpy as np
import tensorflow as tf
import keras
from Data_prep import read_data, partial_format
from ProteinNetworkUtils import Lattice, RemoveMask


def make_short_network(max_aa, lattice_size=7):
    inp = keras.layers.Input(shape=(5, max_aa), dtype=tf.int64)

    # Reshape Inputs
    acids = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    mask = keras.layers.Lambda(lambda x: tf.cast(x[:, 1], tf.bool))(inp)
    aa_length = keras.layers.Lambda(lambda x: tf.count_nonzero(x, axis=1, dtype=tf.int64))(mask)
    indices = keras.layers.Lambda(lambda x: tf.transpose(x[:, 2:5, :], perm=(0, 2, 1)))(inp)

    # Construct Lattice and apply convolutions across time axis
    lattice = LatticeSnake(max_aa, lattice_size)([acids, mask, indices])

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(16, (3, 3, 3), padding="valid"))(lattice)
    conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(32, (3, 3, 3), padding="valid"))(conv)
    conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(64, (3, 3, 3), padding="valid"))(conv)
    conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    conv = keras.layers.Activation('relu')(conv)

    acids = keras.layers.Reshape((max_aa, 1))(acids)
    conv = keras.layers.Reshape((max_aa, 64))(conv)
    conv = keras.layers.concatenate([conv, acids], axis=2)
    reverse = keras.layers.Lambda(lambda x: tf.reverse_sequence(x[0], seq_lengths=tf.cast(x[1], tf.int64), seq_axis=1))([conv, aa_length])

    forward_lstm = keras.layers.LSTM(16, return_sequences=True)(conv)
    backward_lstm = keras.layers.LSTM(16, return_sequences=True)(reverse)
    bi_lstm = keras.layers.concatenate([forward_lstm, backward_lstm], axis=2)

    final = keras.layers.Flatten()(bi_lstm)
    pol_fin = keras.layers.Dense(256, activation='relu')(final)
    pol_fin = keras.layers.Dense(64, activation='relu')(pol_fin)
    pol_fin = keras.layers.Dense(12, activation='relu')(pol_fin)

    final = keras.layers.Dense(256, activation='relu')(final)
    final = keras.layers.Dense(64, activation='relu')(final)
    final = keras.layers.Dense(1, activation=None)(final)

    model = keras.Model(inp, [final, pol_fin])
    return model


def make_big_network(max_aa, lattice_size=7):
    inp = keras.layers.Input(shape=(5, max_aa), dtype=tf.int64)

    # Reshape Inputs
    acids = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    mask = keras.layers.Lambda(lambda x: tf.cast(x[:, 1], tf.bool))(inp)
    indices = keras.layers.Lambda(lambda x: tf.transpose(x[:, 2:5, :], perm=(0, 2, 1)))(inp)

    # Construct Lattice and apply convolutions across time axis
    lattice = LatticeSnake(max_aa, lattice_size)([acids, mask, indices])

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(16, (3, 3, 3)))(lattice)
    conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(32, (3, 3, 3)))(conv)
    conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(64, (3, 3, 3)))(conv)
    conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.Reshape((max_aa, 64))(conv)

    # Apply masked LSTM across lattice features
    conv_lstm = BooleanMask()([conv, mask])
    conv_lstm = keras.layers.LSTM(64, return_sequences=True)(conv_lstm)

    # Apply unmasked BiDirectional LSTM across protein string
    bi_lstm = keras.layers.Reshape((max_aa, 1))(acids)
    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(16, return_sequences=True))(bi_lstm)
    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=True))(bi_lstm)

    # Combine the two feature strings
    combined = keras.layers.concatenate([bi_lstm, conv_lstm], axis=2)

    # Apply Masked LSTM and extract final timestep
    combined_masked_lstm = keras.layers.LSTM(128)(combined)
    combined_masked_lstm = keras.layers.Reshape((1, 128))(combined_masked_lstm)

    # Apply unmasked LSTM and extract last 5 timesteps
    combined_unmasked_lstm = RemoveMask()(combined)
    combined_unmasked_lstm = keras.layers.LSTM(128, return_sequences=True)(combined_unmasked_lstm)
    combined_unmasked_lstm = keras.layers.Lambda(lambda x: x[:, -5:, :])(combined_unmasked_lstm)

    final = keras.layers.concatenate([combined_masked_lstm, combined_unmasked_lstm], axis=1)
    final = keras.layers.Flatten()(final)
    pol_fin = keras.layers.Dense(256, activation='relu')(final)
    pol_fin = keras.layers.Dense(64, activation='relu')(pol_fin)
    pol_fin = keras.layers.Dense(12, activation='relu')(pol_fin)

    final = keras.layers.Dense(256, activation='relu')(final)
    final = keras.layers.Dense(64, activation='relu')(final)
    final = keras.layers.Dense(1, activation=None)(final)

    model = keras.Model(inp, [pol_fin, final])
    return model