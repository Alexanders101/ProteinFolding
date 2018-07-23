from New_arch import make_short_network
import numpy as np
from ProteinEnv import NPProtein
import ProteinNetworkUtils
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

N=48
model = make_short_network(N)
model.compile("adam", loss="MSE", loss_weights=[100, 1])
filename = "/Users/danielstephens/weights_saved_Jul23_epoch_good.h5"
model.load_weights(filename)

NUM_EPOCHS = 1
NUM_SAMPLES = 10
start = NPProtein(N, 1)
for x in range(NUM_EPOCHS):
    count = 0
    y = np.random.randint(20, 30, NUM_SAMPLES)
    store = np.zeros(((N-1)*NUM_SAMPLES, 5, 48), dtype=np.int64)
    policies = np.zeros(((N-1)*NUM_SAMPLES, 6))
    values = np.zeros(((N-1)*NUM_SAMPLES, 1))
    states = np.zeros((NUM_SAMPLES, 5, 48), dtype=np.int64)
    energies = np.zeros(NUM_SAMPLES)
    for z in range(NUM_SAMPLES):
        string = np.zeros(N) + 2
        string[:y[z]] = 1
        string = np.random.permutation(string)
        state = start.new_state(string)
        for u in range(N-1):
            num_moves = list(start.legal(state))
            mid = np.zeros((len(num_moves), 5, 48))
            for t in range(len(num_moves)):
                mid[t] = start.next_state(state, num_moves[t])
            if not num_moves:
                count += 1
                continue
            policy, value = model.predict(mid)
            move = np.argmax(value)
            store[count] = state
            values[count] = value[move]
            policies[count, num_moves[move]] = 1
            state = start.next_state(state, num_moves[move])
            count += 1
        energies[z] = start.reward(state)
    plt.hist(energies)
    print(np.mean(energies))
    filepath="weights_saved_greedy_Jul23_epoch_{}.h5".format(x)
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    #callbacks_list = [checkpoint]
    print("Epoch {}".format(x))
    model.fit(x=store, y=[policies, values])
    model.save_weights(filepath)