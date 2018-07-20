import numpy as np
from numba import jit, int64
from numba.types import Set
import random

from ParallelMCTS import SinglePlayerEnvironment


@jit("int64[::1](int64[:, :], int64)")
def coord_hash(coords, L):
    num_coords = coords.shape[0]
    result = np.empty(num_coords, np.int64)
    for i in range(num_coords):
        coord = coords[i]
        result[i] = coord[0] * L * L * 4 + coord[1] * L * 2 + coord[2]
    return result


@jit('int64(int64[:, ::1], int64)', nopython=True)
def find_end(protein_string, max_size):
    i = 0
    while protein_string[0, i] > 0 and i < max_size:
        i += 1
    return i - 1


@jit('void(int64[:, ::1], int64[::1])', nopython=True)
def _next_state(state, move):
    index = state[1, 0]
    state[1, index] = 1
    previous = state[2:, index - 1]
    state[2:, index] = previous + move
    state[1, 0] = index + 1


@jit('int64[:, :, ::1](int64[:, ::1], int64[:, ::1])', nopython=True)
def _next_state_multi(state, moves):
    num_moves = moves.shape[0]
    result = np.empty((num_moves, state.shape[0], state.shape[1]), dtype=np.int64)
    for i in range(num_moves):
        result[i, :, :] = state[:, :]
        _next_state(result[i], moves[i])
    return result


@jit(Set(int64)(int64[:, ::1], int64[:, ::1]), nopython=True)
def _legal(state, directions):
    size = state.shape[1]
    current_index = state[1, 0] - 1
    last_coord = state[2:, current_index]
    possible_moves = last_coord + directions

    possible_moves_H = coord_hash(possible_moves, size)
    visited_H = set(coord_hash(state[2:, :current_index + 1].T, size))
    result = set()
    for i in range(directions.shape[0]):
        h = possible_moves_H[i]
        if h not in visited_H:
            result.add(i)
    return result


class NPProtein(SinglePlayerEnvironment):
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
        self._state_shape = (5, max_length)

        self.directions = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])
        self._moves = np.arange(0, len(self.directions))
        self.energy_distance = energy_distance
        self._max_length = max_length

    @property
    def moves(self):
        return self._moves

    @property
    def state_shape(self):
        return self._state_shape

    @property
    def max_length(self):
        return self._max_length

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
        _next_state(state, next_move)
        return state

    def next_state_multi(self, state, actions):
        """
        Compute multiple action on a single state. This is used for optimization purposes.

        """
        moves = self.directions[actions]
        return _next_state_multi(state, moves)

    def random_state(self, length=None):
        if length is None:
            length = self.max_length

        random_string = np.random.randint(1, 3, size=length)
        return self.new_state(random_string)

    def random_moves(self, state, length=None, policy=False):
        if length is None:
            length = self.max_length
        if policy:
            choices = np.zeros(length-1)
            big_state = np.zeros((state.shape[1], state.shape[0], state.shape[1]))
        for x in range(length-1):
            potential = list(self.legal(state))
            if not potential:
                state = self.new_state(state[0,:])
                return self.random_moves(state, length, policy=policy)
            choice = random.choice(potential)
            if policy:
                choices[x] = choice
                big_state[x] = state
            state = self.next_state(state, choice)
        if policy:
            big_state[x+1] = state
            choices = choices.astype(int)
            one_hot = np.zeros((length-1, len(self.directions)))
            one_hot[np.arange(length-1), choices] = 1
            return state, big_state, one_hot
        return state

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
        return _legal(state, self.directions)

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
        final_idx = find_end(state, self.max_length)
        return state[1, final_idx] > 0

    def reward(self, state):
        """
        Get the energy value of a given state

        Parameters
        ----------
        state : np.ndarray[int64, (5, max_length)]
            State object

        Returns
        -------
        float [0, inf]
            -energy
        """
        aa_string = state[0, ]
        lattice = state[2:, ].T
        num = int(state[1, 0])
        tot_energy = 0
        for i in range(num):
            for j in range(i + 2, num):
                if aa_string[i] + aa_string[j] == 2 and np.linalg.norm(lattice[i] - lattice[j]) <= self.energy_distance:
                    tot_energy -= 1
        return -tot_energy

    def eval_energy(self, state):
        # New energy evaluation method which is much faster than reward.
        idx = state[2:]
        mask1 = state[0] - 2
        mask2 = state[1]
        idx = idx.astype(np.float64)
        # creating masks to mask out P and 0s.
        mask1 = mask1.astype(np.bool)
        mask2 = mask2.astype(np.bool)
        mask = mask1 & mask2
        idx = idx.T
        na = np.sum(idx ** 2, axis=1)
        # casting as a row and column vectors.
        row = np.reshape(na, [-1, 1])
        col = np.reshape(na, [1, -1])
        # return pairwise euclidead difference matrix.
        result = np.sqrt(row - 2 * np.matmul(idx, idx.T) + col)
        result *= np.tri(*result.shape)
        # masking out the diagonal with offset -1 to prevent comparison of neighboring amino acids.
        np.fill_diagonal(result[1:, :-1], 0)
        result2 = result[mask, :]
        result3 = result2[:, mask]
        final1 = result3 <= self.energy_distance
        final2 = result3 > 0
        return np.sum(final1 & final2)
