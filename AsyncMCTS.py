"""
Author: Alexander Shmakov
Date: 2018-02-13
Version: 0.9

This module contains an implementation of the distribution Monte Carlo Tree Search Algo.

This version of MCTS has a manager class for making predictions to the network (see PredictorThread),
and runs the simulation on K different threads simultaneously for each timestep.

Version Notes
-------------
1.1: Adapted AsyncMCTS to work with Protein Environment

0.9: Implemented a version of virutal loss. However, this virtual loss is currently only calculated
for each edge, not each node. This means its possible to end up at the same state if you take a different
path to it. Ideally, for 1.0, implement a full node-based virtual loss.

0.8: Base implementation. Need to implement virtual loss.

"""
from PredictorThread import PredictorThread
from NetworkProcess import NetworkProcess
from queue import Queue
from threading import Thread, Lock
from time import time

import numpy as np

class AsyncMCTS:
    CONFIG_DEFAULTS = {
        "calculation_time": 15,
        "C": 1.4,
        "batch_size": None,
        "temperature": 1.0,
        "epsilon": 0.3,
        "alpha": 0.03,
        "virtual_loss": 2.0,
        "verbose": 0,
        "preinitialize": True,
        "single_tree": False
    }

    def __init__(self, env, make_model, session_config, num_threads=2, **kwargs):
        """
        Create a Monte Carlo tree search with asynchronous simulation.

        Parameters
        ----------
        env : BaseCube
            The cube control object, should derive from BaseCube or have the same interface
        network : keras.Model
            A keras model containing the policy and value networks. This networks
            mapping is: State -> (Policy, Value)
        session_config : tf.ConfigProto
            A config object for the Tensorflow session created.
        kwargs
            See configuration options

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
        preinitialize : default = True
            Whether or not to intialize an expanded node with the original predicted value
        single_tree : default = False
            Whether or not all of the workers share a single tree
        """
        # Set up environment
        self.env = env
        self.num_moves = len(self.env.moves)
        self.state_shape = self.env.state_shape

        # Run Dynamic Config setup
        self.config(**kwargs)

        # Fix alpha to be an array of alphas
        self.alpha = np.repeat(self.alpha, repeats=self.num_moves)
        self.previous_move = -1

        # Simulation database
        self.N = {}     # Number of visits for each node
        self.W = {}     # Total value of node
        self.Q = {}     # Normalized value of node
        self.V = {}     # Virtual loss count

        # Set up Network
        self.network_process = NetworkProcess(make_model=make_model,
                                              state_shape=self.state_shape,
                                              num_moves=self.num_moves,
                                              num_states=self.num_moves + 1,
                                              num_workers=num_threads,
                                              batch_size=self.batch_size,
                                              session_config=session_config)

        # Multi-threading stuff
        self.num_threads = num_threads
        self.trees = [set() for _ in range(num_threads)]
        self.store_lock = Lock()
        self.num_nodes = 0
        ###########################################################
        self.pred_time = []
        self.run_time = []
        ###########################################################

        self.network_process.start()


    def __del__(self):
        self.network_process.shutdown()

    def __str__(self):
        out = []
        out.append("="*50)
        out.append("MCTS Parameter Values")
        out.append("-"*50 + "\n")
        for name, default_value in self.CONFIG_DEFAULTS.items():
            value = self.__getattribute__(name)
            if name == "alpha":
                value = value[0]

            out.append("{:20} = {:>8} {}".format(name, value, "  --- default" if value == default_value else ""))

        out.append("="*50 + "\n")

        return "\n".join(out)

    def __repr__(self):
        return self.__str__()

    def config(self, **config_opt):
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

    def _add_to_store(self, state_hash):
        """
        Add a node the the N, W, Q, and V dictionaries if it is not already there.
        Initialize them to their 0 values. Perform these steps inside of a Lock since
        "x in dict" is not a thread-safe operation.

        Highly recommended to first test "state_hash in self.N" before calling this
        to avoid having to acquire and release locks all the time.

        Parameters
        ----------
        state_hash : hashable object
            Hashed version of the state. Usually just array.tostring() for number arrays.

        """
        with self.store_lock:
            if state_hash not in self.N:
                self.N[state_hash] = np.zeros(self.num_moves, dtype=np.int32)
                self.W[state_hash] = np.zeros(self.num_moves, dtype=np.float32)
                self.Q[state_hash] = np.zeros(self.num_moves, dtype=np.float32)
                self.V[state_hash] = np.zeros(self.num_moves, dtype=np.int32)


    def _predict_policy_and_value(self, idx, states):
        """
        Asynchronously predict the policies and values of an array of states

        Parameters
        ----------
        idx : int
            Index of calling thread.
        states : ndarray
            Array of states

        Returns
        -------
        (Policies, Values)

        """
        return self.network_process.predict(idx, states)

    def clear(self):
        """
        Clear all of the simulation memory

        """
        self.Q.clear()
        self.W.clear()
        self.N.clear()
        self.V.clear()

    def play(self, start_state, clear=True):
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


    def select_moves(self, state):
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
        # #####
        # Time parameters
        # ###########################
        max_time = self.calculation_time
        episode_start_time = time()

        # #####
        # Main simulation calculation
        # #############################

        # Function for defining a simulation runner
        def runner(idx):
            self.trees[idx].clear()
            start_time = time()
            while (time() - start_time) < max_time:
                self.simulation(idx, state)

        # Track how many nodes we visit per second
        self.num_nodes = 0
        ###########################################################
        self.pred_time.clear()
        self.run_time.clear()
        ###########################################################

        # Create and start the asynchronous workers
        workers = []
        for i in range(self.num_threads):
            thread = Thread(target=runner, args=(i,))
            thread.start()
            workers.append(thread)

        # Wait for all workers to finish
        for thread in workers:
            thread.join()

        if self.verbose >= 2:
            print("{} Nodes per second".format(self.num_nodes / (time() - episode_start_time)))
            ###########################################################
            if len(self.run_time) > 0:
                print("Avg RunTime: {}".format(np.mean(self.run_time)))
                print("Avg PredTime: {}".format(np.mean(self.pred_time)))
            ###########################################################

        # #####
        # Calculate policy distribution
        # ######################
        t = 1 / self.temperature
        pi = self.N[self.env.hash(state)]
        pi = pi ** t + np.finfo(np.float32).eps  # Add tiny value to remove 0s
        pi = pi / pi.sum()

        return pi

    def simulation(self, idx, start_state):
        """
        Perform a single run of the simulation tree, stopping at a leaf node or at the search depth.

        Parameters
        ----------
        idx : int
            index of the thread running this function
        start_state : ndarray
            Root node

        Returns
        -------
        solved : Bool
            Whether or not a solution node was found
        simulation_path : List
            list of the moves that the simulation took

        """
        # #####
        # Storage for simulation run
        # ############################
        simulation_path = []
        state = start_state
        state_hash = self.env.hash(start_state)
        done = False

        if self.single_tree:
            tree = self.trees[0]
        else:
            tree = self.trees[idx]

        # #####
        # Create the necessary data for the root node
        # #############################################
        # next_states = [self.cube.next_state(state, move) for move in self.cube.moves]
        # policy, values = self._predict_policy_and_value(idx, np.asarray(next_states + [state]))
        next_states = np.concatenate((self.env.next_state_multi(state, self.env.moves), np.expand_dims(state, 0)))
        policy, values = self._predict_policy_and_value(idx, next_states)
        policy = policy[-1]
        values = values[:-1, 0]

        if self.verbose >= 3:
            print("Policy: {}".format(policy))
            print("Value: {}".format(values))

        # Add randomness to the initial policy prediction for exploration
        epsilon = self.epsilon
        policy = ((1 - epsilon) * policy) + (epsilon * np.random.dirichlet(self.alpha))

        # #####
        # Evaluate
        # ##########
        ###########################################################
        t0 = time()
        ###########################################################
        while state_hash in tree:
            # Calculate Simulation statistics (From Page 8 of Alpha Go Zero)
            virtual_loss = self.V.get(state_hash, 0) * self.virtual_loss
            N = self.N[state_hash]
            U = self.C * policy * np.sqrt(N.sum() + 1) / (1 + N)
            A = U + self.Q[state_hash] - virtual_loss

            if self.verbose >= 3:
                print("Q: {}".format(self.Q[state_hash]))
                print("N: {}".format(N))
                print("U: {}".format(U))
                print("A: {}".format(A))

            # Get the best valid move for the current state
            sorted_actions = np.argsort(A)
            best_action_idx = None
            legal_choices = self.env.legal(state)
            for possible_action in reversed(sorted_actions):
                if possible_action in legal_choices:
                    best_action_idx = possible_action
                    break

            if best_action_idx is None:
                if self.verbose >= 1:
                    print("Dead End Found")
                break

            # Get the state of our action and the predicted reward
            next_state = next_states[best_action_idx]
            last_value = values[best_action_idx]
            self.V[state_hash][best_action_idx] += 1
            self.num_nodes += 1

            # Update loop variables
            simulation_path.append((state_hash, best_action_idx))
            state = next_state
            state_hash = self.env.hash(state)

            # If we reach the end of the tree, break out.
            if self.env.done(state):
                done = True
                break

            # next_states = [self.env.next_state(state, move) for move in self.env.moves]
            # policy, values = self._predict_policy_and_value(idx, np.asarray(next_states + [state]))
            next_states = np.concatenate((self.env.next_state_multi(state, self.env.moves), np.expand_dims(state, 0)))
            ###########################################################
            self.run_time.append(time() - t0)
            t0 = time()
            ###########################################################
            policy, values = self._predict_policy_and_value(idx, next_states)
            ###########################################################
            self.pred_time.append(time() - t0)
            t0 = time()
            ###########################################################
            policy = policy[-1]
            values = values[:-1, 0]

        # Initialize new node
        if state_hash not in self.N:
            self._add_to_store(state_hash)
        tree.add(state_hash)

        # #####
        # Backup
        # ########
        # if self.N[state_hash][0] == 0 and self.preinitialize:
        #     self.N[state_hash] += 1
        #     self.W[state_hash] += values
        #     self.Q[state_hash] = self.W[state_hash] / self.N[state_hash]

        for hash, action in simulation_path:
            self.N[hash][action] += 1
            self.W[hash][action] += last_value
            # self.Q[hash][action] = self.W[hash][action] / self.N[hash][action]
            self.Q[hash][action] = max(self.Q[hash][action], last_value)
            self.V[hash][action] -= 1

        return done, simulation_path
