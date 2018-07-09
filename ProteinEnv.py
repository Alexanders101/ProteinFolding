import numpy as np
from numba import jit, vectorize, guvectorize, int64
import xxhash

@guvectorize([(int64[:], int64, int64[:])], "(n),()->()", nopython=True, target="cpu")
def coord_hash(coord, L, res):
    res[0] = coord[0] * L * L * 4 + coord[1] * L * 2 + coord[2]


@jit('int64(int64[:, ::1], int64)', nopython=True)
def find_end(protein_string, max_size):
    i = 0
    while protein_string[0, i] > 0 and i < max_size:
        i += 1
    return i - 1


class NPProtein():
    def __init__(self, max_length, energy_distance=2):
        """
        Container Class for NP Protein Environment.

        Parameters
        ----------
        max_length : int
            Maximum length of protein strings
        energy_distance : float
            Distance to evaluate neighboring acids
        """
        self.moves = np.arange(0, 12)
        self.directions = np.array([(1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)])
        self.energy_distance = energy_distance

        self.max_length = max_length
        self.state_shape = (5, max_length)

    def new_state(self, protein_string):
        """
        Create a new NP Protein state from an amino acid sequence

        Parameters
        ----------
        protein_string : np.ndarray[int64]
            Numpy array of amino acid sequence. H = 1, P = 2.

        Returns
        -------
        np.ndarray[int64, (5, max_length)]
            State representation of protein.

        """
        protein_length = protein_string.shape[0]
        assert protein_length <= self.max_length, "Input protein is longer than maximum allowed protein"

        out = np.zeros((5, self.max_length), dtype=np.int64)
        out[0, :protein_length] = protein_string
        out[1, 0] = 1

        return out

    def next_state(self, state, action):
        """
        Compute the next protein configuration given the direction to place the next protein.

        Parameters
        ----------
        state : np.ndarray[int64, (5, max_length)]
            State object
        action : int
            One of the elements in self.moves

        Returns
        -------
        np.ndarray[int64, (5, max_length)]
            A new state object
        """
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
        """
        Compute multiple action on a single state. This is used for optimization purposes.

        """
        return np.asarray([self.next_state(state, x) for x in actions])
        # save = self.legal(state)
        # return [self.next_state(state, x) for x in actions if x in save]

    def random_state(self, length=None):
        if length is None:
            length = self.max_length

        random_string = np.random.randint(1, 3, size=length)
        return self.new_state(random_string)

    def legal(self, state):
        """
        The valid moves to perform given a state.
        Parameters
        ----------
        state : np.ndarray[int64, (5, max_length)]
            State object

        Returns
        -------
        set
            The valid moves that can be performed.

        """
        size = state.shape[1]
        current_index = state[1, 0] - 1
        last_coord = state[2:, current_index]
        possible_moves = last_coord + self.directions

        possible_moves_H = coord_hash(possible_moves, size)
        visited_H = set(coord_hash(state[2:, :current_index + 1].T, size))

        return {i for i, h in enumerate(possible_moves_H) if h not in visited_H}

    def hash(self, state):
        """
        Convert a state object in a hashable datatype for use in python dicts.
        Parameters
        ----------
        state : np.ndarray[int64, (5, max_length)]
            State object

        Returns
        -------
        Hashed Value

        """
        # return xxhash.xxh64(state, seed=0).intdigest()
        return state.tostring()

    def done(self, state):
        """
        Whether or not a given state is an ending state.

        Parameters
        ----------
        state : np.ndarray[int64, (5, max_length)]
            State object

        Returns
        -------
        bool

        """
        # return state[1, -1] > 0
        final_idx = find_end(state, self.max_length)
        return state[1, final_idx] > 0

    def reward(self, state):
        """
        Get the reward value of a given state

        Parameters
        ----------
        state : np.ndarray[int64, (5, max_length)]
            State object

        Returns
        -------
        float [0, inf]
            Reward
        """
        aa_string = state[0, ]
        lattice = state[2:, ].T
        num = state[1, 0]
        tot_energy = 0
        for i in range(num):
            for j in range(i + 2, num):
                if aa_string[i] + aa_string[j] == 2 and np.linalg.norm(lattice[i] - lattice[j]) <= self.energy_distance:
                    tot_energy -= 1
        return -tot_energy
