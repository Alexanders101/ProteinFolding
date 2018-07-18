import numpy as np
import tensorflow as tf

from tensorflow import keras
from ProteinNetworkUtils import RemoveMask, BooleanMask, LatticeSnake, eval_energy

distr_48=np.array([[0,0,0,0,0,0,0,1,1,2,2,3,4,4,5,6,7,8,10,10,12,13,14,16,17,19,20,22,24,26,27,28,31,33,35,37,40,42,44,47,50,52,54,58,60,64,65,62],
[0,0,0,0,0,0,1,1,1,2,3,3,4,5,6,6,8,9,10,11,12,13,15,16,18,20,21,23,25,26,28,30,32,35,37,39,41,44,47,49,51,55,58,60,64,65,62,0],
[0,0,0,0,0,0,1,1,1,2,3,4,4,5,6,7,8,9,10,12,13,14,15,17,19,20,22,25,26,27,29,31,34,36,38,41,43,46,48,51,54,57,59,64,65,65,0,0],
[0,0,0,0,0,0,1,1,2,2,3,4,4,5,6,7,8,9,11,12,13,15,16,18,20,21,23,25,26,29,30,33,35,37,40,42,45,48,49,53,56,59,63,64,67,0,0,0],
[0,0,0,0,0,0,1,1,2,2,3,4,4,5,6,7,8,10,11,12,13,15,16,18,20,21,24,25,27,29,31,34,36,39,40,44,47,48,52,55,57,62,63,67,0,0,0,0],
[0,0,0,0,0,0,1,1,2,2,3,4,5,5,7,7,9,10,11,12,14,15,17,19,20,23,24,26,28,30,33,34,37,39,42,45,47,50,54,56,61,62,66,0,0,0,0,0],
[0,0,0,0,0,0,1,1,2,3,3,4,5,6,7,8,9,10,12,13,14,16,18,19,21,23,24,27,29,31,33,36,38,41,44,46,48,52,55,59,61,64,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,2,3,3,4,5,6,7,8,9,11,12,13,15,17,18,20,22,24,26,28,30,32,35,37,39,43,45,47,51,53,59,60,63,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,2,3,3,4,5,6,7,8,10,11,12,14,16,17,19,21,22,24,26,29,31,33,36,38,41,44,46,50,51,57,58,61,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,2,3,4,4,5,6,7,9,10,11,13,15,16,18,19,21,23,25,28,29,32,35,36,40,43,44,48,50,55,57,60,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,1,2,3,4,4,6,6,8,9,10,12,13,15,16,18,20,22,23,27,28,30,34,35,38,41,43,47,49,53,56,59,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,2,2,3,4,5,6,7,8,9,11,12,14,15,17,18,21,22,25,27,29,32,34,37,40,41,45,48,52,54,57,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,1,2,2,3,4,5,6,7,8,10,11,13,14,16,17,19,21,23,26,28,30,33,35,39,41,43,47,50,53,56,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,2,2,3,4,5,6,7,8,10,12,13,14,16,18,20,22,25,26,29,31,34,37,39,42,45,49,51,54,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,2,3,3,4,5,6,8,9,10,12,13,15,17,19,21,23,25,27,30,32,35,38,40,44,48,49,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,2,3,3,4,5,6,8,9,11,12,14,16,17,20,22,24,26,28,31,34,36,39,42,46,48,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,2,3,3,5,5,7,8,10,11,13,15,16,18,20,23,24,27,29,32,35,37,40,44,47,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,2,3,4,5,6,7,8,10,11,13,15,16,19,21,23,26,28,31,33,35,39,43,45,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,1,2,3,4,5,6,7,9,10,12,14,15,17,20,22,24,26,29,32,34,37,41,44,48,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,2,2,3,4,5,6,8,9,11,13,14,16,18,20,23,25,28,30,32,36,39,43,46,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,2,2,3,4,5,7,8,10,11,13,15,16,19,21,23,26,28,31,34,38,41,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,2,2,3,4,6,7,8,10,11,14,15,17,20,22,25,27,30,33,36,39,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,1,2,3,3,5,6,8,9,10,12,14,16,18,20,23,25,28,31,34,38,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,2,3,4,5,6,8,9,11,13,14,17,19,21,24,27,29,33,36,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,2,3,4,5,6,8,10,11,13,15,18,20,23,25,28,31,34,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,2,3,4,5,7,8,10,12,14,16,18,21,24,26,30,33,36,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,2,3,4,6,7,9,11,12,14,17,19,22,25,28,31,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,2,3,5,6,8,9,11,13,16,18,20,23,26,29,33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,1,2,4,5,6,8,10,12,14,17,19,21,25,28,31,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,2,3,4,5,7,9,10,13,15,17,20,23,26,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,2,3,4,6,7,9,11,13,16,18,21,25,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,2,3,4,6,8,10,12,14,17,20,23,26,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,2,3,5,6,9,10,12,15,18,21,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,1,2,4,5,7,9,11,14,16,19,23,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,1,3,4,6,8,10,12,15,18,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,2,3,4,6,8,10,13,16,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,2,3,5,7,9,11,14,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,2,4,5,7,10,12,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,2,4,6,8,11,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,1,3,4,7,9,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,2,3,5,8,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,2,4,6,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,2,4,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,1,3,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,2,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

def make_short_network(max_aa, lattice_size=7):
    inp = keras.layers.Input(shape=(5, max_aa), dtype=tf.int64)

    # Reshape Inputs
    acids = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    mask = keras.layers.Lambda(lambda x: tf.cast(x[:, 1], tf.bool))(inp)
    aa_length = keras.layers.Lambda(lambda x: tf.count_nonzero(x, axis=1, dtype=tf.int64))(mask)
    indices = keras.layers.Lambda(lambda x: tf.transpose(x[:, 2:5, :], perm=(0, 2, 1)))(inp)
    current = keras.layers.Lambda(lambda x: x[:, 1, 0])(inp)
    num_left = keras.layers.Lambda(lambda x: tf.map_fn(lambda x: tf.count_nonzero(x[x[0]+1:]), x[current:]-2))

    # Construct Lattice and apply convolutions across time axis
    lattice = LatticeSnake(max_aa, lattice_size)([acids, mask, indices])

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(64, (5, 5, 5), padding="valid"))(lattice)
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

    #conv = keras.layers.TimeDistributed(keras.layers.Conv3D(16, (3, 3, 3)))(lattice)
    #conv = keras.layers.TimeDistributed(keras.layers.BatchNormalization())(conv)
    #conv = keras.layers.Activation('relu')(conv)

    conv = keras.layers.TimeDistributed(keras.layers.Conv3D(32, (3, 3, 3)))(lattice)
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

def model_not(max_aa):
    inp = keras.layers.Input(shape=(5, max_aa), dtype=tf.int64)
    acids = keras.layers.Lambda(lambda x: tf.cast(x[:, 0], tf.float32))(inp)
    current = keras.layers.Lambda(lambda x: tf.expand_dims(x[:, 1, 0]-1, 1))(inp)

    num_left = tf.expand_dims(keras.layers.Lambda(lambda x: tf.map_fn(lambda x: max_aa - x[0] - tf.count_nonzero(x[x[0] + 1:] - 1), x))(
            tf.concat([current, tf.cast(acids, tf.int64)], axis=1)), 1)

    hueristic_energy_left = keras.layers.Lambda(lambda x: tf.expand_dims(tf.gather_nd(distr_48, x), 1))(tf.concat([current, num_left], axis=1))

    current_energy = keras.layers.Lambda(lambda x: tf.expand_dims(tf.map_fn(eval_energy, x), 1))(inp)
    predicted_energy = keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(tf.concat([tf.cast(hueristic_energy_left, tf.int64), current_energy], axis=1))

    policy = keras.layers.Lambda(lambda x: 0 * tf.reduce_sum(x, axis=1)[:, :12] + 1)(inp)
    # policy = keras.layers.Lambda(lambda x: tf.multiply(x, tf.constant(0, dtype=tf.int64)))(policy)
    # policy = keras.layers.Lambda(lambda x: tf.add(x, tf.constant(1, dtype=tf.int64)))(policy)

    return keras.Model(inp, [policy, predicted_energy])
