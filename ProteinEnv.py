import numpy as np
from numba import jit, vectorize

@jit("int64(int64[::1])", nopython=True)
def last_nonzero(arr):
    size = arr.shape[0]
    for i in range(size):
        if arr[i] == 0:
            return i-1
    return -1

class NPProtein():
    def __init__(self):
        self.moves = np.arange(0, 6)

    def new_state(self, protein_string):
        protein_length = protein_string.shape[0]

        out = np.zeros((5, protein_length), dtype=np.int64)
        out[0, :] = protein_string
        out[1, 0] = 1

        return out

    def next_state(self, state, action):
        pass

    def next_state_multi(self, state, actions):
        pass

    def legal(self, state, move):
        pass

    def hash(self, state):
        pass

    def done(self, state):
        pass

    def reset(self):
        pass
