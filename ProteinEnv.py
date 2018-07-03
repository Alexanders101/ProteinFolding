import numpy as np
from numba import jit, vectorize, guvectorize, int64
import xxhash

@guvectorize([(int64[:], int64, int64[:])], "(n),()->()", nopython=True, target="cpu")
def coord_hash(coord, L, res):
    res[0] = coord[0] * L * L * 4 + coord[1] * L * 2 + coord[2]

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
        next_move = self.directions[action]
        index = state[1, 0]
        state[1, index] = 1
        previous = state[2:, index - 1]
        next = previous + next_move
        state[2:, index] = next
        state[1, 0] = index + 1
        return state

    def next_state_multi(self, state, actions):
        pass

    def legal(self, state):
        L = state.shape[1]
        current_index = state[1, 0] - 1
        last_coord = state[2:, current_index]
        possible_moves = last_coord + self.directions

        possible_moves_H = coord_hash(possible_moves, L)
        visited_H = set(coord_hash(state[2:, :current_index + 1].T, L))

        return {i for i, h in enumerate(possible_moves_H) if h not in visited_H}

    def hash(self, state):
        return xxhash.xxh64(state).intdigest()

    def done(self, state):
        pass

    def reset(self):
        pass
