import numpy as np
from numba import jit, vectorize, guvectorize, int64
import xxhash

@guvectorize([(int64[:], int64, int64[:])], "(n),()->()", nopython=True, target="cpu")
def coord_hash(coord, L, res):
    res[0] = coord[0] * L * L * 4 + coord[1] * L * 2 + coord[2]

class NPProtein():
    def __init__(self, energy_distance=2):
        self.moves = np.arange(0, 12)
        self.directions = np.array([(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)])
        self.energy_distance = energy_distance

    def new_state(self, protein_string):
        protein_length = protein_string.shape[0]

        out = np.zeros((5, protein_length), dtype=np.int64)
        out[0, :] = protein_string
        out[1, 0] = 1

        return out

    def next_state(self, state, action):
        state = state.copy()
        next_move = self.directions[action]
        index = state[1, 0]
        state[1, index] = 1
        previous = state[2:, index - 1]
        next = previous + next_move
        state[2:, index] = next
        state[1, 0] = index + 1
        return state

    def next_state_multi(self, state, actions):
        return [self.next_state(state, x) for x in actions]
        # save = self.legal(state)
        # return [self.next_state(state, x) for x in actions if x in save]

    def legal(self, state):
        size = state.shape[1]
        current_index = state[1, 0] - 1
        last_coord = state[2:, current_index]
        possible_moves = last_coord + self.directions

        possible_moves_H = coord_hash(possible_moves, size)
        visited_H = set(coord_hash(state[2:, :current_index + 1].T, size))

        return {i for i, h in enumerate(possible_moves_H) if h not in visited_H}

    def hash(self, state):
        return xxhash.xxh64(state, seed=0).intdigest()
        #return state.tostring()

    def done(self, state):
        return state[1, -1] > 0

    def reward(self, state):
        aa_string = state[0, ]
        lattice = state[2:, ].T
        num = state[1, 0]
        tot_energy = 0
        for i in range(num):
            for j in range(i + 2, num):
                if aa_string[i] + aa_string[j] == 2 and np.linalg.norm(lattice[i] - lattice[j]) <= self.energy_distance:
                    tot_energy -= 1
        return -tot_energy
