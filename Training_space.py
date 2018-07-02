import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy
import sparse
from Data_prep import read_data, partial_format

filename = "/Users/danielstephens/PycharmProjects/Proteinfolding/venv/small_data_partial.h5"
save, Y, sizes = read_data(filename)

acids, lat_map, loc, Ys = partial_format(save, sizes, Y)

max_size = 25
max_aa = 40

inp1 = keras.layers.Input(shape=(max_aa,))
inp3 = keras.layers.Input(shape=(max_aa,))
inp2 = keras.layers.Input(shape=(2*max_size-1, 2*max_size-1, 2*max_size-1, 1))

mid = keras.layers.Conv3D(16, (4, 4, 4))(inp2)
mid = keras.layers.BatchNormalization()(mid)
mid = keras.layers.MaxPooling3D()(mid)
mid = keras.layers.Activation('relu')(mid)

mid = keras.layers.Conv3D(32, (4, 4, 4))(mid)
mid = keras.layers.BatchNormalization()(mid)
mid = keras.layers.MaxPooling3D()(mid)
mid = keras.layers.Activation('relu')(mid)
mid = keras.layers.Flatten()(mid)

temp = keras.layers.Dense(50)(inp1)
temp = keras.layers.Dropout(.5)(temp)
temp = keras.layers.Activation('relu')(temp)

merge = keras.layers.concatenate([mid, temp, inp3])

out = keras.layers.Dense(64, activation="relu")(merge)
last = keras.layers.Dense(1, activation="linear")(out)
model = keras.Model(inputs=[inp1, inp2, inp3], outputs=last)

model.compile(loss='mean_squared_error', optimizer='Adam', metrics=['mae'])

model.fit([acids, lat_map, loc], Ys, batch_size=64, epochs=1, verbose=1)

#score = model.evaluate([acids_test, lat_map_test], test_Y)