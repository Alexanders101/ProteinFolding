from os import environ
environ['CUDA_VISIBLE_DEVICES'] = ''

from Arch2D import make_short_network_2D_greedy
import numpy as np
from ProteinEnv import NPProtein
from ProteinNetworkUtils import Lattice2D
from tensorflow import keras
import tensorflow as tf

CONFIG = tf.ConfigProto(device_count = {'GPU': 0}, 
                        log_device_placement=True, 
                        allow_soft_placement=True,
                        intra_op_parallelism_threads=32,
                        inter_op_parallelism_threads=32)
CONFIG.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=CONFIG)
keras.backend.set_session(sess)

N=20
model = make_short_network_2D_greedy(N)
model.compile("adam", loss=["categorical_crossentropy", "MSE"], loss_weights=[10, 1])
#filename = "/Users/danielstephens/weights_saved_Jul23_epoch_good.h5"
#model.load_weights(filename)

determined_values = True

NUM_EPOCHS = 1000
NUM_SAMPLES = 25
start = NPProtein(N, 1, 2)
threshold = 2


def diver(state, depth, model):
    if state[1, 0] == state.shape[1] - 1:
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
            nothin, mid[t] = diver(start.next_state(state, num_moves[t]), depth - 1, model)
        idx = np.argmax(mid)
        return num_moves[idx], mid[idx]


for x in range(NUM_EPOCHS):
    mistakes = 0
    count = 0
    count2 = 0
    y = np.random.randint(11, 12, NUM_SAMPLES)
    store = np.zeros(((N - 1) * NUM_SAMPLES, 4, N), dtype=np.int64)
    energies = np.zeros((NUM_SAMPLES, 1))
    policies = np.zeros(((N - 1) * NUM_SAMPLES, 4))
    values = np.zeros(((N - 1) * NUM_SAMPLES, 1))
    while count2 < NUM_SAMPLES:
        string = np.zeros(N) + 2
        string[:y[count2]] = 1
        string = np.random.permutation(string)
        state = start.new_state(string)
        for u in range(N - 1):
            move, value = diver(state, 2, model)
            if move == -10 or value == -10:
                count = count2 * (N - 1)
                break
            store[count] = state
            state = start.next_state(state, move)
            if not determined_values:
                if u != N - 2:
                    values[count] = value
                else:
                    values[count] = start.eval_energy(state)
            policies[count, move] = 1
            count += 1
        energy = start.eval_energy(state)
        if energy < threshold:
            count = count2 * (N - 1)
            mistakes += 1
        else:
            energies[count2] = energy
            count2 += 1
    print(np.mean(energies), np.max(energies))
    print("# of discarded folds : {}".format(mistakes))
    threshold = np.percentile(energies, 50)
    if determined_values:
        values = np.repeat(energies + 1, N - 1)
    filepath = "saved_weights_20/weights_saved_greedy_Jul31_epoch0_{}.h5".format(x)
    # checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
    # callbacks_list = [checkpoint]
    print("Epoch {}".format(x))
    model.fit(x=store, y=[policies, values])
    model.save_weights(filepath)
