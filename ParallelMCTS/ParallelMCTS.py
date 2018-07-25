"""
Author: Alexander Shmakov
Date: 2018-02-13
Version: 1.1

This module contains an implementation of the distribution Monte Carlo Tree Search Algorithm.

Version Notes
-------------
1.1: Adapted AsyncMCTS to work with Protein Environment

0.9: Implemented a version of virutal loss. However, this virtual loss is currently only calculated
for each edge, not each node. This means its possible to end up at the same state if you take a different
path to it. Ideally, for 1.0, implement a full node-based virtual loss.

0.8: Base implementation. Need to implement virtual loss.

"""
from ParallelMCTS.NetworkManager import NetworkManager
from ParallelMCTS.DistributedNetworkProcess import DistributedNetworkConfig
from ParallelMCTS.DataProcess import DataProcess
from ParallelMCTS.SimulationProcess import SimulationProcessManager
from ParallelMCTS.SinglePlayerEnvironment import SinglePlayerEnvironment

import numpy as np
import tensorflow as tf
from typing import Callable, Tuple
from time import time


class ParallelMCTS:
    CONFIG_DEFAULTS = {
        "calculation_time": 15,
        "C": 1.4,
        "batch_size": None,
        "temperature": 1.0,
        "epsilon": 0.3,
        "alpha": 0.03,
        "virtual_loss": 2.0,
        "verbose": 0,
        "single_tree": False,
        "backup_true_value": False
    }

    def __init__(self, env: SinglePlayerEnvironment, make_model: Callable[[], tf.keras.Model],
                 num_workers: int = 2, num_networks: int = 4, session_config: tf.ConfigProto = None, *,
                 simulation_manager=SimulationProcessManager, simulation_options: dict = {},
                 network_options: dict = {}, database_options: dict = {}, **kwargs):
        """
        Create a Monte Carlo tree search with asynchronous simulation.

        Parameters
        ----------
        env : Subclass of SinglePlayerEnvironment
            An environment object defining how to plat your game.
        make_model : () -> keras.Model
            A function defining how to create your model.
            The resulting network has the following signature: State -> (Policy, Value)
        num_workers : int
            Number of simulation workers for a single game.
        num_networks : int
            Number of prediction networks.
        session_config : tf.ConfigProto
            A config object for the Tensorflow session created.
        simulation_manager : SimulationProcessManager
            Class for performing simulations. Only override this if you know what you are doing.
        simulation_options : dict
            Extra options for simulation manager.
        network_options : dict
            Extra options to pass to NetworkManager. ParallelMCTS.NetworkOptions() provides all options with defaults.
        database_options : dict
            Extra options to pass to DataProcess. ParallelMCTS.DatabaseOptions() provides all options with defaults.
        kwargs
            See configuration options below. ParallelMCTS.MCTSOptions() provides all options with defaults.

        Config Options
        --------------
        calculation_time : default = 15
            Number of seconds to allow for selecting a move
        C : default = 1.4
            Exploration parameter for the node selection
        batch_size : default = None
            Batch size for prediction. Defaults to num_threads * num_moves for optimality.
        temperature : default = 1.0
            Exploration parameter for move selection
        epsilon : default = 0.3
            Randomness factor of root node policy prediction
        alpha : default = 0.01
            Parameter of dirichlet distribution
        virtual_loss : default = 2
            Multiple of visit counts to subtract from value for every visit to a given state
            during a given simulation
        verbose : default = 0
            Verbosity of output
        single_tree : default = False
            Whether or not all of the workers share a single tree
        backup_true_value : default = False
            Whether or not the backup uses the final predicted value or the final true value
        """
        # Set up environment
        self.env = env
        self.num_moves = len(self.env.moves)
        self.state_shape = self.env.state_shape

        # Run Dynamic Config setup
        self.set_config(**kwargs)

        # Fix alpha to be an array of alphas
        self.alpha = np.repeat(self.alpha, repeats=self.num_moves)

        # Simulation database
        self.database = DataProcess(self.num_moves, num_workers, single_tree=self.single_tree, **database_options)

        # Setup Networks
        self.network_manager = NetworkManager(make_network=make_model,
                                              state_shape=self.state_shape,
                                              num_moves=self.num_moves,
                                              num_states=1,
                                              num_workers=num_workers,
                                              batch_size=self.batch_size,
                                              num_networks=num_networks,
                                              session_config=session_config,
                                              **network_options)

        # Setup Simulation Workers
        self.workers = simulation_manager(num_workers, env, self.network_manager, self.database, self.get_config(),
                                          **simulation_options)

    def start(self):
        print("Starting Networks")
        self.network_manager.start()
        self.network_manager.wait_until_all_ready()

        print("Starting Database")
        self.database.start()

        print("Starting Workers")
        self.workers.start()

    def shutdown(self):
        print("Shutting Down Workers")
        self.workers.shutdown()
        print("Shutting Down Network Manager")
        self.network_manager.shutdown()
        print("Shutting Down Database")
        self.database.shutdown()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def __str__(self):
        out = []
        out.append("="*50)
        out.append("MCTS Parameter Values")
        out.append("-"*50 + "\n")
        for name, default_value in self.CONFIG_DEFAULTS.items():
            value = self.__getattribute__(name)
            if name == "alpha":
                value = value[0]

            out.append("{:20} = {:>8} {}".format(name, "None" if value is None else value,
                                                 "  --- default" if value == default_value else ""))

        out.append("="*50 + "\n")

        return "\n".join(out)

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def MCTSOptions() -> dict:
        return ParallelMCTS.CONFIG_DEFAULTS.copy()

    @staticmethod
    def NetworkOptions() -> dict:
        options = dict(zip(DistributedNetworkConfig.__init__.__code__.co_varnames[1:-1],
                           DistributedNetworkConfig.__init__.__defaults__))
        options['train_buffer_size'] = 64
        options['start_port'] = 2222
        options['num_ps'] = None
        return options

    @staticmethod
    def DatabaseOptions() -> dict:
        options = {'synchronous': True, 'num_action_threads': 16}
        return options

    def set_config(self, **config_opt) -> None:
        """
        Create Optional Configuration variables for the Class.

        The parameter names and defaults should be defined in the CONFIG_DEFAULTS dictionary.

        See constructor docstring for options

        Parameters
        ----------
        config_opt : CONFIG_DEFAULTS

        """
        for name, default in self.CONFIG_DEFAULTS.items():
            if name in config_opt:
                self.__setattr__(name, config_opt[name])
            elif name not in self.__dict__:
                self.__setattr__(name, default)

    def get_config(self) -> dict:
        """ Get a dictionary of the current config values. """
        out = {}
        for name in self.CONFIG_DEFAULTS:
            out[name] = self.__getattribute__(name)
        return out

    def clear(self) -> None:
        """ Clear all of the simulation memory. """
        self.database.clear()

    def fit_epoch(self, num_games: int = 1) -> None:
        """ Play a certain number of games and then train on the resulting data.

        Parameters
        ----------
        num_games : int
            Number of games to play for data collection.
        """
        states = []
        policies = []
        values = []

        for game in range(num_games):
            start_state = self.env.random_state()
            s, pi, r = self.play(start_state)

            states.append(s)
            policies.append(pi)
            values.append(r)

        states = np.concatenate(states)
        policies = np.concatenate(policies)
        values = np.concatenate(values)

        self.network_manager.fit(states, policies, values)

    def play(self, start_state: np.ndarray, clear: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Play a full episode and return training data.

        Parameters
        ----------
        start_state : np.ndarray
            Starting state for the episode.

        clear : bool
            Whether or not to clear MCTS databases before playing. This should be TRUE
            during training and FALSE during playing.

        Returns
        -------
        states : np.ndarray[T, ...STATE_SHAPE]
            States visited during episode
        PIs : np.ndarray[T, NUM_MOVES]
            Calculated policies for each state
        R : np.ndarray[T, 1]
            Final reward of episode. This is identical for all states, simplified form for easier training.

        Notes
        -----
        The size of the return values, T, will be based on how long the given protein is. It will only return those
        values where it had information, T can be anywhere from [1, N], with N being the maximum length of the protein.
        """
        if clear:
            self.clear()

        state = start_state  # Current state of the game, this will be updated during play

        # #####
        # Return Buffers
        # ################
        states = np.zeros(shape=(self.env.max_length - 1, *self.env.state_shape), dtype=np.int64)
        pis = np.zeros(shape=(self.env.max_length - 1, self.env.moves.shape[0]), dtype=np.float32)

        # #####
        # Play game until finished
        # ###########################
        t = 0
        while not self.env.done(state):
            pi = self.select_moves(state)

            states[t, :] = state
            pis[t, :] = pi
            t += 1

            next_move = np.random.choice(np.arange(0, len(pi)), p=pi)
            state = self.env.next_state(state, next_move)

        # Value target is final reward of episode
        R = np.repeat(self.env.reward(state), t)
        R = np.expand_dims(R, 1)

        # Training Data
        return states[:t], pis[:t], R

    def select_moves(self, state: np.ndarray) -> np.ndarray:
        """
        Calculate a policy from a starting state using MCTS.

        Parameters
        ----------
        state : ndarray
            Starting state.

        Returns
        -------
        PI : np.ndarray
            A distribution over possible moves determining which one will be optimal from the given state.

        """
        episode_start_time = time()

        # Start simulation on new state
        self.workers.set_start_state(state)
        self.workers.simulation(clear_tree=True)

        # Block and wait for results
        results = self.workers.results()

        if self.verbose >= 2:
            print("{} Nodes per second".format(np.sum(results['num_nodes']) / (time() - episode_start_time)))
            print("Avg RunTime: {}".format(np.mean(results['mean_run_time'])))
            print("Avg PredTime: {}".format(np.mean(results['mean_pred_time'])))

        # #####
        # Calculate policy distribution
        # ######################
        t = 1 / self.temperature
        pi = self.database.get(0, self.env.hash(state))[0]
        pi = pi ** t + np.finfo(np.float32).eps  # Add tiny value to remove 0s
        pi = pi / pi.sum()

        return pi