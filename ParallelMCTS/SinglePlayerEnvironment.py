import numpy as np

from abc import ABC, abstractmethod
from typing import Set, List, Hashable

class SinglePlayerEnvironment(ABC):
    """ This is the abstract base class of any environment that is used with ParallelMCTS. The environment consists
    of the following methods, which operate on a state object, in the form of a numpy array, in a memory-free manner.

    All of the information about a state must be stored in a single numpy array. These states are passed around
    efficient inside of MCTS."""

    @property
    @abstractmethod
    def moves(self) -> np.ndarray:
        """ This property is a numpy array representing the possible moves in your environment. """
        pass

    @property
    @abstractmethod
    def state_shape(self) -> List[int]:
        """ This property returns the static shape of any state in the environment. """
        pass

    @abstractmethod
    def new_state(self, state_definition: object) -> np.ndarray:
        """ This method is responsible for create a new state object from some simpler definition.

        A state must be represented by a single numpy array with a constant shape defined by state_shape.

        Parameters
        ----------
        state_definition : object
            Any state definition you wish to use.

        Returns
        -------
        numpy array
            A new state object.
        """
        pass

    @abstractmethod
    def next_state(self, state: np.ndarray, action: object) -> np.ndarray:
        """ Perform an action on a given state and return the resulting state.

        Parameters
        ----------
        state : numpy array
            The state to act on.
        action : object
            One of the actions in your moves property.

        Returns
        -------
        numpy array
            The next state.
        """
        pass

    def next_state_multi(self, state: np.ndarray, actions: [object]) -> [np.ndarray]:
        """ An extension of the next_state method that allows multiple actions on a single state. The output
        will be the result of applying each action in actions to the state.

        In the most simplest case, it simply loops over the actions and calls next state on each one.
        If your state allows you to perform many different actions in parallel in some efficient way, then you
        can overwrite this class to provide better performance.

        Parameters
        ----------
        state : numpy array
            The state to act on.
        actions : [object]
            A list of action objects.

        Returns
        -------
        [numpy array]
            A list of the resulting states.
        """
        return [self.next_state(state, action) for action in actions]

    @abstractmethod
    def legal(self, state: np.ndarray) -> Set[object]:
        """ The legal moves to perform from a given state.

        Parameters
        ----------
        state : numpy array
            The state in question.

        Returns
        -------
        Set[object]
            A set of move objects filled with moves that are valid from this state.

        """
        pass

    @abstractmethod
    def done(self, state: np.ndarray) -> bool:
        """ Determine whether or not a state is the end of an episode.

        IE, after this state, there are no more possible moves.

        Parameters
        ----------
        state : numpy array
            The state in question.

        Returns
        -------
        bool
            Whether or not the game is over.
        """
        pass

    @abstractmethod
    def reward(self, state) -> float:
        """ The reward value for the state.

        Parameters
        ----------
        state : numpy array
            The state in question.

        Returns
        -------
        float
            A real number representing the value of this state.
        """
        pass

    def hash(self, state: np.ndarray) -> Hashable:
        """ Since numpy arrays are not hashable by default, this method provides a way of converting a numpy array
        to a unique hash value.

        Usually, you may leave this as the default, which return the bytestring representation of the state. However,
        if your state supports a more efficient way of representing it as a hashable type (such as a number or string),
        feel free to overwrite this method.

        Parameters
        ----------
        state : numpy array
            The state in question.

        Returns
        -------
        Hashable
            A python object implementing the __hash__ method.
        """
        return state.tostring()

    def random_state(self, *args) -> np.ndarray:
        """ Optional method to generate a random state. This is useful for not  having to manually generate
        a random new_state. """
        raise NotImplementedError()
