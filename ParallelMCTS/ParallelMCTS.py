"""
Author: Alexander Shmakov
Date: 2018-02-13
Version: 1.1

This module contains an implementation of the distribution Monte Carlo Tree Search Algorithm.

Version Notes
-------------
1.1: Adapted AsyncMCTS to work with Protein Environment

0.9: Implemented a version of virtual loss. However, this virtual loss is currently only calculated
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
from time import sleep
from concurrent.futures import ThreadPoolExecutor

from multiprocessing import cpu_count

# noinspection PyPep8Naming
class ParallelMCTS:
    CONFIG_DEFAULTS = {
        "calculation_time": 15.0,
        "C": 1.4,
        "temperature": 1.0,
        "epsilon": 0.3,
        "alpha": 0.03,
        "virtual_loss": 2.0,
        "verbose": 0,
        "single_tree": False,
        "backup_true_value": False
    }

    # noinspection PyDefaultArgument
    def __init__(self, env: SinglePlayerEnvironment, make_model: Callable[[], tf.keras.Model],
                 num_parallel: int = 1, num_workers: int = 2, num_networks: int = 4,
                 session_config: tf.ConfigProto = None, *,
                 simulation_manager: type = SimulationProcessManager, simulation_options: dict = {},
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
        num_parallel : int
            Number of maximum parallel games to play.
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
        self.num_moves = self.env.num_moves
        self.state_shape = tuple(self.env.state_shape)

        # Class parameters
        self.num_parallel = num_parallel
        self.num_workers = num_workers
        self.num_networks = num_networks
        self.num_total_workers = num_workers * num_parallel

        # Run Dynamic Config setup
        self.set_config(**kwargs)

        # Fix alpha to be an array of alphas
        self.alpha = np.repeat(self.alpha, repeats=self.num_moves)

        # Setup Network Manager. There is only one manager that will be shared across all parallel simulations.
        self.network_manager = NetworkManager(make_network=make_model,
                                              state_shape=self.state_shape,
                                              state_type=env.state_type,
                                              num_moves=env.num_moves,
                                              num_states=1,
                                              num_workers=self.num_total_workers,
                                              num_networks=self.num_networks,
                                              session_config=session_config,
                                              **network_options)

        # Setup Simulation Workers and Database
        self.databases = []
        self.workers = []
        for idx in range(num_parallel):
            database = DataProcess(self.num_moves, num_workers, single_tree=self.single_tree, **database_options)
            worker = simulation_manager(idx, num_workers, env, self.network_manager, database, self.get_config(),
                                        **simulation_options)
            self.databases.append(database)
            self.workers.append(worker)

        # Thread pool for multi-train
        self.worker_thread_pool = ThreadPoolExecutor(max_workers=num_parallel)

    # region ControlMethods
    def start(self, training_network_wait_time: float = 3):
        print("Starting Networks")
        print("="*60)
        self.network_manager.start(wait_time=training_network_wait_time)
        self.network_manager.wait_until_all_ready()

        print("\nStarting Databases")
        print("=" * 60)
        for i, database in enumerate(self.databases):
            print("Starting Database {}".format(i))
            database.start()

        sleep(1)
        print("\nStarting Workers")
        print("=" * 60)
        for i, worker in enumerate(self.workers):
            print("Starting Worker {}".format(i))
            worker.start()

    def shutdown(self):
        print("Shutting Down Workers")
        print("=" * 60)
        for i, worker in enumerate(self.workers):
            print("Shutting Down Worker {}".format(i))
            worker.shutdown()

        print("\nShutting Down Network Manager")
        print("=" * 60)
        self.network_manager.shutdown()

        print("\nShutting Down Databases")
        print("=" * 60)
        for i, database in enumerate(self.databases):
            print("Shutting Down Database {}".format(i))
            database.shutdown()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
    # endregion

    # region PrintingMethods
    # noinspection PyListCreation
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
    # endregion

    # region StaticOptionMethods
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
        options['batch_size'] = None
        return options

    @staticmethod
    def DatabaseOptions() -> dict:
        options = {'synchronous': True, 'num_action_threads': 16}
        return options

    @staticmethod
    def GenerateTensorflowConfig(num_networks: int = 4,
                                 num_gpu: int = 1,
                                 growth: bool = False,
                                 gpu_memory_ratio: float = 0.95) -> tf.ConfigProto:
        num_networks += 1

        num_cpu = cpu_count()

        optimizer_options = tf.OptimizerOptions(do_common_subexpression_elimination=True,
                                                do_constant_folding=True,
                                                do_function_inlining=True,
                                                opt_level=tf.OptimizerOptions.L1)

        graph_options = tf.GraphOptions(optimizer_options=optimizer_options)

        gpu_options = tf.GPUOptions(allow_growth=growth)

        config = tf.ConfigProto(device_count={'GPU': num_gpu},
                                allow_soft_placement=True,
                                intra_op_parallelism_threads=num_cpu,
                                inter_op_parallelism_threads=num_cpu,
                                graph_options=graph_options,
                                gpu_options=gpu_options)

        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

        return config
    # endregion

    # region ConfigMethods
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
    # endregion

    def clear(self, idx: int) -> None:
        """ Clear all of the simulation memory.

        Parameters
        ----------
        idx : int
            Which simulation worker's database to clear.
        """
        self.databases[idx].clear()

    def _fit_epoch_multi_work(self, worker_idx: int, num_games: int, states, policies, values):
        for game in range(num_games):
            start_state = self.env.random_state()
            s, pi, r = self.play(worker_idx, start_state, clear=True)

            states.append(s)
            policies.append(pi)
            values.append(r)

    def fit_epoch_multi(self, num_games_per_worker: int = 1):
        states = []
        policies = []
        values = []

        futures = []
        for worker_idx in range(self.num_parallel):
            future = self.worker_thread_pool.submit(self._fit_epoch_multi_work, worker_idx, num_games_per_worker,
                                                    states, policies, values)
            futures.append(future)

        for future in futures:
            future.result()

        states = np.concatenate(states)
        policies = np.concatenate(policies)
        values = np.concatenate(values)

        self.network_manager.fit(states, policies, values)

    def fit_epoch_single(self, num_games: int = 1, worker_idx: int = 0) -> None:
        """ Play a certain number of games and then train on the resulting data.

        This is the single-process version. Only one simulation worker will be launched.

        Parameters
        ----------
        num_games : int
            Number of games to play for data collection.
        worker_idx : int
            Which worker should play
        """
        states = []
        policies = []
        values = []

        for game in range(num_games):
            start_state = self.env.random_state()
            s, pi, r = self.play(worker_idx, start_state, clear=True)

            states.append(s)
            policies.append(pi)
            values.append(r)

        states = np.concatenate(states)
        policies = np.concatenate(policies)
        values = np.concatenate(values)

        self.network_manager.fit(states, policies, values)

    def play(self, worker_idx: int, start_state: np.ndarray, clear: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Play a full episode and return training data.

        Parameters
        ----------
        worker_idx : int
            Index of this simulation worker.
        start_state : np.ndarray
            Starting state for the episode.
        clear : bool = True
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
            self.clear(worker_idx)

        # Current state of the game, this will be updated during play
        state = start_state.copy()

        # Return Buffers
        states = np.zeros(shape=(self.env.max_length, *self.env.state_shape), dtype=self.env.state_type)
        pis = np.zeros(shape=(self.env.max_length, self.env.num_moves), dtype=np.float32)

        # Play game until finished
        t = 0
        while not self.env.done(state):
            # Calculate legal moves and bail if we're at a dead end
            legal_moves = self.env.legal(state)
            if len(legal_moves) == 0:
                break

            # Run simulation to compute policy
            pi = self.select_moves(worker_idx, state)
            
            # Store game data in buffer
            states[t, :] = state
            pis[t, :] = pi
            t += 1

            # Set all illegal moves Probability to 0
            for move_idx in range(self.num_moves):
                if move_idx not in legal_moves:
                    pi[move_idx] = -np.inf
            
            # Softmax
            pi = np.exp(pi)
            pi /= np.sum(pi)
            
            # Sample from policy and make next move
            next_move = np.random.choice(pi.shape[0], p=pi)
            state = self.env.next_state(state, self.env.moves[next_move])

        # Value target is final reward of episode
        R = np.repeat(self.env.reward(state) - self.env.num_moves + t, t)
        R = np.expand_dims(R, 1)
        R = R.astype(np.float32)

        # Training Data
        return states[:t], pis[:t], R

    def _temperature_policy(self, N: np.ndarray) -> np.ndarray:
        """ Calculate Alpha-Go Zero policy from visit counts.

        Parameters
        ----------
        N : np.ndarray
            Visit counts of root node.

        Returns
        -------
        PI : np.ndarray
            A distribution over possible moves determining which one will be optimal from the given state.
        """
        t = 1 / self.temperature
        pi = N ** t + np.finfo(np.float32).eps  # Add tiny value to remove 0s
        pi = pi / pi.sum()
        return pi

    def select_moves_async(self, worker_idx: int, state: np.ndarray) -> None:
        """
        Calculate a policy from a starting state using MCTS.

        Parameters
        ----------
        worker_idx : int
            Index of this simulation worker.
        state : ndarray
            Starting state.
        """
        worker = self.workers[worker_idx]

        # Start simulation on new state
        worker.set_start_state(state)
        worker.simulation(clear_tree=True)

    def select_moves_result(self, worker_idx: int) -> np.ndarray:
        """ Get result of an async select_moves call.

        Behaviour is not defined if select_moves_async was not called prior.

        Parameters
        ----------
        worker_idx : int
            Index of this simulation worker.

        Returns
        -------
        PI : np.ndarray
            A distribution over possible moves determining which one will be optimal from the given state.
        """
        worker = self.workers[worker_idx]
        database = self.databases[worker_idx]
        state = worker.starting_state.copy()

        # Block and wait for results
        results = worker.results()

        if self.verbose >= 2:
            print("{} Nodes per second".format(np.sum(results['total_nodes']) / self.calculation_time))

        return self._temperature_policy(database.get(0, self.env.hash(state))[0])

    def select_moves(self, worker_idx: int, state: np.ndarray) -> np.ndarray:
        """
        Calculate a policy from a starting state using MCTS.

        Parameters
        ----------
        worker_idx : int
            Index of this simulation worker.
        state : ndarray
            Starting state.

        Returns
        -------
        PI : np.ndarray
            A distribution over possible moves determining which one will be optimal from the given state.

        """
        worker = self.workers[worker_idx]
        database = self.databases[worker_idx]

        # Start simulation on new state
        worker.set_start_state(state)
        worker.simulation(clear_tree=True)

        # Block and wait for results
        results = worker.results()

        if self.verbose >= 2:
            print("{} Nodes per second".format(np.sum(results['total_nodes']) / self.calculation_time))

        return self._temperature_policy(database.get(0, self.env.hash(state))[0])
