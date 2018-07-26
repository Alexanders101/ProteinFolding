from multiprocessing import Process, Array, Event, Value
from ParallelMCTS.NetworkManager import NetworkManager
from ParallelMCTS.DataProcess import DataProcess
from ParallelMCTS.SinglePlayerEnvironment import SinglePlayerEnvironment, py2c
import numpy as np
import ctypes
import os
import signal
from time import time


# noinspection PyAttributeOutsideInit,PyPep8Naming
class SimulationProcess(Process):
    def __init__(self, idx: int, network_offset: int, env: SinglePlayerEnvironment,
                 network_manager: NetworkManager, database: DataProcess,
                 state_buffer_base: Array, state_shape: tuple, mcts_config: dict):
        super(SimulationProcess, self).__init__()

        self.idx = idx
        self.env = env
        self.database = database
        self.network_manager = network_manager
        self.network_offset = network_offset

        self.starting_state = np.ctypeslib.as_array(state_buffer_base)
        self.starting_state = self.starting_state.reshape(state_shape)

        self.input_queue = Event()
        self.output_queue = Event()
        self.output_queue.set()

        self.input_param = Value('l', 0, lock=False)
        self.output_num_nodes = Value('d', 0, lock=False)

        self.calculation_time = mcts_config['calculation_time']
        self.C = mcts_config['C']
        self.epsilon = mcts_config['epsilon']
        self.alpha = mcts_config['alpha']
        self.virtual_loss = mcts_config['virtual_loss']
        self.verbose = mcts_config['verbose']
        self.backup_true_value = mcts_config['backup_true_value']

    def shutdown(self) -> None:
        """ Shutdown this simulation server. Blocks until simulation server is free.
        """
        self.output_queue.wait()
        self.output_queue.clear()

        self.input_param.value = -1
        self.input_queue.set()

    def simulation(self, clear_tree: bool = True) -> None:
        """ Queue a single simulation run.

        Parameters
        ----------
        clear_tree : bool = True
            Whether or not to clear the simulation tree for this simulation.
            This is useful if you want to split the simulation into multiple batches.
        """
        self.output_queue.wait()
        self.output_queue.clear()

        if clear_tree:
            self.input_param.value = 0
        else:
            self.input_param.value = 1

        self.input_queue.set()

    def result(self) -> dict:
        """ Block until simulation has finished and get the result dictionary.

        Returns
        -------
        result_dict : dict
            Dictionary of various simulation statistics.
        """
        self.output_queue.wait()
        return {'nodes_per_second': self.output_num_nodes.value}

    def _predict_single_node(self, idx: int, state: np.ndarray):
        """ Use network to predict on a single state.

        Parameters
        ----------
        idx : int
            Worker index.
        state : np.ndarray
            A NON-BATCHED single state.

        Returns
        -------
        policy : np.ndarray
            The policy prediction for this state.
        value : float
            The value prediction for this state.
        """
        policy, value = self.network_manager.predict(self.network_offset + idx, np.expand_dims(state, 0))
        return policy[0], value[0, 0]

    # noinspection PyUnboundLocalVariable
    def _simulation(self, idx: int):
        """ Perform a single simulation run until encountering a leaf node.

        Parameters
        ----------
        idx : int
            Worker index.
        """
        simulation_path = []
        state = self.starting_state.copy()
        state_hash = self.env.hash(self.starting_state)

        # Create the necessary data for the root node
        # noinspection PyTupleAssignmentBalance
        not_leaf_node, data = self.database.both_get(idx, state_hash)
        policy = self.root_policy
        if data is not None:
            N, W, Q, V, _ = data

        last_value = None
        done = False

        # Loop until leaf node.
        while not_leaf_node and data is not None:
            # Calculate Simulation statistics (From Page 8 of Alpha Go Zero)
            virtual_loss = V * self.virtual_loss
            U = self.C * policy * np.sqrt(N.sum()) / (1 + N)
            A = U + Q - virtual_loss

            if self.verbose >= 3:
                print("Q: {}".format(Q))
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

            # Bail if we have encountered a dead end
            if best_action_idx is None:
                if self.verbose >= 1:
                    print("Dead End Found")
                break

            # Get result of action and add a visit count to database
            next_state = self.env.next_state(state, self.env.moves[best_action_idx])
            self.database.visit(state_hash, best_action_idx)
            self.num_nodes += 1

            # Update loop variables
            simulation_path.append((state_hash, best_action_idx))
            state = next_state
            state_hash = self.env.hash(state)

            # If we reach the end of the tree, break out.
            if self.env.done(state):
                done = True
                if self.backup_true_value:
                    last_value = self.env.reward(state)
                break

            # Get data and policy cache for the next node
            # noinspection PyTupleAssignmentBalance
            not_leaf_node, data = self.database.both_get(idx, state_hash)
            if data is not None:
                N, W, Q, V, policy = data

        # Extra Processing done by subclasses.
        self._process_paths(idx, done, last_value, simulation_path)

        # Initialize new node
        policy, value = self._predict_single_node(idx, state)
        self.database.both_add(idx, state_hash, policy)
        if last_value is None:
            last_value = value

        # Backup all nodes on path
        for state_hash, action in simulation_path:
            self.database.backup(state_hash, action, last_value)

    def _run_simulation(self, idx, command) -> None:
        """ Start an MCTS simulation for the configured time.

        Parameters
        ----------
        idx : int
            Worker index.
        command : int
            Command received.
        """
        # Clear search tree for new simulation.
        if command == 0:
            self.database.tree_clear(idx)

        # Cache root node policy and add randomization. Add the root node to the tree for small optimization.
        self.root_policy, self.root_value = self._predict_single_node(idx, self.starting_state.copy())
        self.root_policy = ((1 - self.epsilon) * self.root_policy) + (self.epsilon * np.random.dirichlet(self.alpha))
        self.database.both_add(idx, self.env.hash(self.starting_state), self.root_policy)

        # Run simulations until we run out of time.
        start_time = time()
        while (time() - start_time) < self.calculation_time:
            self._simulation(idx)

    def _process_paths(self, idx: int, done: bool, last_value: float, simulation_path) -> None:
        """ Overwrite this method in specializations of SimulationProcess.

        This is called before the backup step in order to perform any
        extra processing on the MCTS path.

        Parameters
        ----------
        idx : int
            Index of worker.
        done : bool
            Whether or not eh simulation concluded on an ending state.
        last_value : float or None
            The value of the final state. This could be None.
        simulation_path : [(hash, move)]
            The path MCTS took during that simulation.
        """
        pass

    def run(self):
        idx = self.idx
        self.num_nodes = 0

        while True:
            self.input_queue.wait()
            self.input_queue.clear()
            command = self.input_param.value

            if command == -1:
                break

            t0 = time()
            self._run_simulation(idx, command)
            t1 = time()
            
            self.output_num_nodes.value = self.num_nodes / (t1 - t0)
            self.output_queue.set()
            self.num_nodes = 0


class SimulationProcessManager:
    def __init__(self, server_index: int, num_workers: int, env: SinglePlayerEnvironment,
                 network_manager: NetworkManager, database: DataProcess, mcts_config: dict,
                 *, worker_class: type = SimulationProcess, **worker_args):
        self.server_index = server_index
        self.num_workers = num_workers
        self.env = env

        self.network_manager = network_manager
        self.database = database
        self.mcts_config = mcts_config

        self.state_shape = self.env.state_shape
        state_type = py2c(env.state_type)
        self.state_buffer_base = Array(state_type, int(np.prod(env.state_shape)), lock=False)
        self.starting_state = np.ctypeslib.as_array(self.state_buffer_base)
        self.starting_state = self.starting_state.reshape(env.state_shape)

        network_offset = server_index * num_workers
        self.workers = []
        for idx in range(num_workers):
            worker = worker_class(idx, network_offset, env, network_manager, database,
                                  self.state_buffer_base, self.state_shape, mcts_config, **worker_args)
            self.workers.append(worker)

    def start(self) -> None:
        """ Start all workers. """
        for worker in self.workers:
            worker.start()

    def shutdown(self) -> None:
        """ Kill workers without mercy. """
        print("Killing Worker:", end="")
        for idx, worker in enumerate(self.workers):
            if worker.pid:
                print(" {}".format(idx), end="")
                os.kill(worker.pid, signal.SIGKILL)
        print()

    def set_start_state(self, state: np.ndarray) -> None:
        """ Set the starting state for the simulation workers.

        Parameters
        ----------
        state: np.ndarray
            Starting state.
        """
        self.starting_state[:] = state.copy()

    def simulation(self, clear_tree: bool = True) -> None:
        """ Being a single simulation run for all workers.

        Parameters
        ----------
        clear_tree : bool
            Whether or not to clear the current game tree before starting the simulation.
        """
        for worker in self.workers:
            worker.simulation(clear_tree)

    def results(self):
        res = [worker.result() for worker in self.workers]
        return {k: [dic[k] for dic in res] for k in res[0]}


def __simulation_old(self, idx: int):
    simulation_path = []
    state = self.starting_state.copy()
    state_hash = self.env.hash(self.starting_state)

    # #####
    # Create the necessary data for the root node
    # #############################################
    next_states = np.concatenate((self.env.next_state_multi(state, self.env.moves), np.expand_dims(state, 0)))
    policy, values = self.network_manager._predict_unsafe(idx, next_states)
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
    done = False
    last_value = None
    not_leaf_node, data = self.database.both_get(idx, state_hash)
    while not_leaf_node:
        # Calculate Simulation statistics (From Page 8 of Alpha Go Zero)
        N, _, Q, V = data
        virtual_loss = V * self.virtual_loss
        U = self.C * policy * np.sqrt(N.sum() + 1) / (1 + N)
        A = U + Q - virtual_loss

        if self.verbose >= 3:
            print("Q: {}".format(Q))
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
        self.database.visit(state_hash, best_action_idx)
        self.num_nodes += 1

        # Update loop variables
        simulation_path.append((state_hash, best_action_idx))
        state = next_state
        state_hash = self.env.hash(state)

        # If we reach the end of the tree, break out.
        if self.env.done(state):
            done = True
            if self.backup_true_value:
                last_value = self.env.reward(state)
            break

        next_states = np.concatenate((self.env.next_state_multi(state, self.env.moves), np.expand_dims(state, 0)))

        ###########################################################
        self.run_time.append(time() - t0)
        t0 = time()
        ###########################################################

        policy, values = self.network_manager._predict_unsafe(idx, next_states)

        ###########################################################
        self.pred_time.append(time() - t0)
        t0 = time()
        ###########################################################

        policy = policy[-1]
        values = values[:-1, 0]
        not_leaf_node, data = self.database.both_get(idx, state_hash)

    # Extra Processing done by subclasses.
    self._process_paths(idx, done, last_value, simulation_path)

    # Initialize new node
    self.database.both_add(idx, state_hash)

    # Backup all nodes on path
    for hash, action in simulation_path:
        self.database.backup(hash, action, last_value)

