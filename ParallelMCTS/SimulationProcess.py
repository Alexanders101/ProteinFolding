from multiprocessing import Process, Array, Event, Value, Barrier
from ParallelMCTS.NetworkManager import NetworkManager
from ParallelMCTS.DataProcess import DataProcess
from ParallelMCTS.SinglePlayerEnvironment import SinglePlayerEnvironment, np2c
import numpy as np
import os
import signal
from time import time


# noinspection PyAttributeOutsideInit,PyPep8Naming
class SimulationProcess(Process):
    def __init__(self, idx: int, network_offset: int, env: SinglePlayerEnvironment,
                 network_manager: NetworkManager, database: DataProcess, state_buffer_base: Array,
                 starting_depth: Value, state_shape: tuple, simulation_barrier: Barrier, mcts_config: dict):
        super(SimulationProcess, self).__init__()

        self.idx = idx
        self.env = env
        self.database = database
        self.network_manager = network_manager
        self.network_offset = network_offset
        self.network_idx = network_offset + idx

        self.starting_state = np.ctypeslib.as_array(state_buffer_base)
        self.starting_state = self.starting_state.reshape(state_shape)
        self.starting_depth = starting_depth

        self.simulation_barrier = simulation_barrier
        self.input_ready = Event()
        self.output_ready = Event()
        self.output_ready.set()

        self.input_param = Value('l', 0, lock=False)
        self.output_num_nodes = Value('l', 0, lock=False)

        self.single_tree = mcts_config['single_tree']
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
        self.output_ready.wait()
        self.output_ready.clear()

        self.input_param.value = -1
        self.input_ready.set()

    def simulation(self, clear_tree: bool = True) -> None:
        """ Queue a single simulation run.

        Parameters
        ----------
        clear_tree : bool = True
            Whether or not to clear the simulation tree for this simulation.
            This is useful if you want to split the simulation into multiple batches.
        """
        self.output_ready.wait()
        self.output_ready.clear()

        if clear_tree:
            self.input_param.value = 0
        else:
            self.input_param.value = 1

        self.input_ready.set()

    def result(self) -> dict:
        """ Block until simulation has finished and get the result dictionary.

        Returns
        -------
        result_dict : dict
            Dictionary of various simulation statistics.
        """
        self.output_ready.wait()
        return {'total_nodes': self.output_num_nodes.value}

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
        if data is not None:
            (N, W, Q, V, _) = data
        policy = self.root_policy

        # Local Data
        num_moves = self.starting_depth.value
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
            legal_choices = self.env.legal(state)
            sorted_actions = np.argsort(A)
            best_action_idx = None

            for possible_action in reversed(sorted_actions):
                if possible_action in legal_choices:
                    best_action_idx = possible_action
                    break

            # Bail if we have encountered a dead end
            if best_action_idx is None:
                # Calculate the value of the dead end to be its true value minus the number of moves remaining.
                last_value = self.env.reward(state) - self.env.max_length + num_moves
                break

            # Update databases after visiting this node.
            simulation_path.append((state_hash, best_action_idx))
            self.database.visit(state_hash, best_action_idx)
            self.num_nodes += 1

            # Take the simulated step.
            state = self.env.next_state(state, self.env.moves[best_action_idx])
            state_hash = self.env.hash(state)
            num_moves += 1

            # If we reach the end of the game, break out.
            if self.env.done(state):
                done = True
                if self.backup_true_value:
                    last_value = self.env.reward(state)
                break

            # Otherwise predict on the new state and repeat.
            else:
                # noinspection PyTupleAssignmentBalance
                not_leaf_node, data = self.database.both_get(idx, state_hash)
                if not_leaf_node:
                    N, W, Q, V, policy = data

        # Extra Processing done by subclasses.
        self._process_paths(idx, done, last_value, simulation_path)

        # Initialize new node
        policy, value = self.network_manager.predict_single(self.network_idx, state)
        self.database.both_add(idx, state_hash, policy)
        if last_value is None:
            last_value = value

        # Backup all nodes on path
        for state_hash, action in simulation_path:
            self.database.backup(state_hash, action, last_value)

    def _run_simulation(self, idx: int) -> None:
        """ Start an MCTS simulation for the configured time.

        Parameters
        ----------
        idx : int
            Worker index.
        """
        self.num_nodes = 0
        
        self.root_policy, self.root_value = self.network_manager.predict_single(self.network_idx, self.starting_state.copy())
        self.root_policy = ((1 - self.epsilon) * self.root_policy) + (self.epsilon * np.random.dirichlet(self.alpha))
        # Cache root node policy and add randomization. Add the root node to the tree for small optimization.
        if not self.single_tree or idx == 0:
            self.database.both_add(idx, self.env.hash(self.starting_state), self.root_policy)

        # Synchronize all of the workers so the databases are in sync.
        self.simulation_barrier.wait()

        # Run simulations until we run out of time.
        if self.calculation_time > 0:
            start_time = time()
            while (time() - start_time) < self.calculation_time:
                self._simulation(idx)

        # Run simulations until we achieve the desired number of nodes.
        else:
            max_nodes = -self.calculation_time
            while self.num_nodes < max_nodes:
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
        self.num_nodes = 0

        while True:
            # Wait for new input to be ready
            self.input_ready.wait()
            command = self.input_param.value
            self.input_ready.clear()

            # Exit command
            if command == -1:
                break

            # Clear search tree for new simulation.
            if command == 0:
                self.database.tree_clear(self.idx)

            self._run_simulation(self.idx)

            self.output_num_nodes.value = self.num_nodes
            self.output_ready.set()


class SimulationProcessManager:
    def __init__(self, manager_index: int, num_workers: int, env: SinglePlayerEnvironment,
                 network_manager: NetworkManager, database: DataProcess, mcts_config: dict,
                 *, worker_class: type = SimulationProcess, **worker_args):
        self.manager_index = manager_index
        self.num_workers = num_workers
        self.env = env

        self.network_manager = network_manager
        self.database = database
        self.mcts_config = mcts_config

        state_shape = env.state_shape
        state_type = np2c(env.state_type)
        self.state_buffer_base = Array(state_type, int(np.prod(state_shape)), lock=False)
        self.starting_state = np.ctypeslib.as_array(self.state_buffer_base)
        self.starting_state = self.starting_state.reshape(state_shape)
        self.starting_depth = Value("l", 0, lock=False)

        self.simulation_barrier = Barrier(num_workers)

        network_offset = manager_index * num_workers
        self.workers = []
        for worker_idx in range(num_workers):
            worker = worker_class(worker_idx, network_offset, env, network_manager, database,
                                  self.state_buffer_base, self.starting_depth, state_shape, self.simulation_barrier,
                                  mcts_config, **worker_args)
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
                os.kill(worker.pid, signal.SIGTERM)
        print()

    def set_start_state(self, state: np.ndarray, starting_depth: int = 0) -> None:
        """ Set the starting state for the simulation workers.

        Parameters
        ----------
        state: np.ndarray
            Starting state.
        starting_depth: int
            How deep the start state is in the tree.
        """
        self.starting_state[:] = state.copy()
        self.starting_depth.value = starting_depth

    def simulation(self, clear_tree: bool = True) -> None:
        """ Begin a single simulation run for all workers.

        Parameters
        ----------
        clear_tree : bool
            Whether or not to clear the current game tree before starting the simulation.
        """
        for worker in self.workers:
            worker.simulation(clear_tree)

    def results(self) -> dict:
        """ Get results of every worker. This method blocks until all workers are finished.

        Returns
        -------
        results: dict
            Dictionary where the keys are the result names and the values are lists of results from each worker.
        """
        res = [worker.result() for worker in self.workers]
        return {k: [dic[k] for dic in res] for k in res[0]}
