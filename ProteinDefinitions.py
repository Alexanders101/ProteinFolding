import tensorflow as tf
from tensorflow import keras
from ProteinEnv import NPProtein

# CONFIG OPTIONS
N = 20


def make_model():
    inp = keras.Input((4, N))

    net = keras.layers.Flatten()(inp)

    policy = keras.layers.Dense(4)(net)
    value = keras.layers.Dense(1)(net)
    return keras.Model(inp, [policy, value])


def make_env():
    return NPProtein(N, 1, 2)
