from New_arch import make_short_network
import numpy as np
from ProteinEnv import NPProtein
import ProteinNetworkUtils
from tensorflow import keras
import tensorflow as tf

N = 48
model = make_short_network(N)
model.compile("adam", loss="MSE", loss_weights=[100, 1])
filename = "where_weights_saved"
model.load_weights(filename)

NUM_EPOCHS = 100
NUM_SAMPLES = 100
start = NPProtein(N, 1)
for x in range(NUM_EPOCHS):
    count = 0
    y = np.random.randint(20, 30, NUM_SAMPLES)
    store = np.zeros(((N - 1) * NUM_SAMPLES, 5, 48), dtype=np.int64)
    policies = np.zeros(((N - 1) * NUM_SAMPLES, 6))
    values = np.zeros(((N - 1) * NUM_SAMPLES, 1))
    states = np.zeros((NUM_SAMPLES, 5, 48), dtype=np.int64)
    for z in range(NUM_SAMPLES):
        string = np.zeros(N) + 2
        string[:y[z]] = 1
        string = np.random.permutation(string)
        states[z] = start.new_state(string)
    for u in range(N - 1):
        policy, value = model.predict(states)
        move = np.argmax(policy, axis=1)
        store[count:count + NUM_SAMPLES] = states
        policies[count:count + NUM_SAMPLES] = policy
        values[count:count + NUM_SAMPLES] = value
        for w in range(NUM_SAMPLES):
            states[w] = start.next_state(states[w], move[w])
        count += NUM_SAMPLES
    filepath = "weights_saved_greedy_Jul23_epoch_{}.h5".format(x)
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False,
                                                 mode='max')
    callbacks_list = [checkpoint]
    print("Epoch {}".format(x))
    model.fit(x=store, y=[policies, values], callbacks=callbacks_list)
