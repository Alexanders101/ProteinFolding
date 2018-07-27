from Arch2D import make_short_network_2D
import numpy as np
from ProteinEnv import NPProtein
from ProteinNetworkUtils import Lattice2D
from tensorflow import keras
import tensorflow as tf

N=48
model = make_short_network_2D(N)
model.compile("adam", loss="MSE", loss_weights=[50, 1])
#filename = "/Users/danielstephens/weights_saved_Jul23_epoch_good.h5"
#model.load_weights(filename)

determined_values = True

def diver(protein_env, state, depth, model):
    if state[1,0] == state.shape[1]-1:
        depth = 1
    if depth == 1:
        num_moves = list(start.legal(state))
        if not len(num_moves):
            return -10, -10
        mid = np.zeros((len(num_moves), state.shape[0], state.shape[1]))
        for t in range(len(num_moves)):
            mid[t] = start.next_state(state, num_moves[t])
        policy, value = model.predict(mid)
        max_ = np.max(value)
        return num_moves[np.argmax(value)], max_
    else:
        num_moves = list(start.legal(state))
        if not len(num_moves):
            return -10, -10
        mid = np.zeros(len(num_moves))
        for t in range(len(num_moves)):
            nothin, mid[t] = diver(start, start.next_state(state, num_moves[t]), depth-1, model)
        idx = np.argmax(mid)
        return num_moves[idx], mid[idx]

NUM_EPOCHS = 200
NUM_SAMPLES = 50
start = NPProtein(N, 1, 2)
for x in range(NUM_EPOCHS):
    count = 0
    count2 = 0
    y = np.random.randint(20, 30, NUM_SAMPLES)
    store = np.zeros(((N-1)*NUM_SAMPLES, 4, N), dtype=np.int64)
    energies = np.zeros((NUM_SAMPLES, 1))
    policies = np.zeros(((N-1)*NUM_SAMPLES, 4))
    values = np.zeros(((N-1)*NUM_SAMPLES, 1))
    while count2 < NUM_SAMPLES:
        string = np.zeros(N) + 2
        string[:y[count2]] = 1
        string = np.random.permutation(string)
        state = start.new_state(string)
        for u in range(N-1):
            move, value = diver(start, state, 2, model)
            if move == -10 and value == -10:
                count = count2*(N-1)
                break
            store[count] = state
            state = start.next_state(state, move)
            if not determined_values:
                if u != N-2:
                    values[count] = value
                else:
                    values[count] = start.eval_energy(state)
            policies[count, move] = 1
            count += 1
        energies[count2] = start.eval_energy(state)
        count2 += 1
    print(np.mean(energies), np.max(energies))
    if determined_values:
        values = np.repeat(energies, N-1) + 1
    filepath="weights_saved_greedy_Jul27_epoch_{}.h5".format(x)
    #checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    #callbacks_list = [checkpoint]
    print("Epoch {}".format(x))
    model.fit(x=store, y=[policies, values])
    model.save_weights(filepath)