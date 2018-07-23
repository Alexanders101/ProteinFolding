from multiprocessing import Process, Array, Event, Value
from ParallelMCTS.NetworkManager import NetworkManager
from ParallelMCTS.DataProcess import DataProcess
import numpy as np
import ctypes
import os
import signal
from time import time


class SimulationProcess(Process):
    def __init__(self, idx: int, env, network_manager: NetworkManager, database: DataProcess,
                 state_buffer_base: Array, state_shape: tuple, mcts_config: dict):
        super(SimulationProcess, self).__init__()

        self.idx = idx
        self.env = env
        self.database = database
        self.network_manager = network_manager

        self.starting_state = np.ctypeslib.as_array(state_buffer_base)
        self.starting_state = self.starting_state.reshape(state_shape)

        self.input_queue = Event()
        self.output_queue = Event()
        self.output_queue.set()

        self.input_param = Value('l', 0, lock=False)
        self.output_num_nodes = Value('l', 0, lock=False)
        self.output_pred_time = Value('d', 0.0, lock=False)
        self.output_run_time = Value('d', 0.0, lock=False)

        self.calculation_time = mcts_config['calculation_time']
        self.C = mcts_config['C']
        self.epsilon = mcts_config['epsilon']
        self.alpha = mcts_config['alpha']
        self.virtual_loss = mcts_config['virtual_loss']
        self.verbose = mcts_config['verbose']
        self.backup_true_value = mcts_config['backup_true_value']

    def shutdown(self):
        self.output_queue.wait()
        self.output_queue.clear()

        self.input_param.value = -1
        self.input_queue.set()

    def simulation(self, clear_tree: bool = True):
        self.output_queue.wait()
        self.output_queue.clear()

        if clear_tree:
            self.input_param.value = 0
        else:
            self.input_param.value = 1

        self.input_queue.set()

    def result(self):
        self.output_queue.wait()
        return {'num_nodes': self.output_num_nodes.value,
                'mean_pred_time': self.output_pred_time.value,
                'mean_run_time': self.output_run_time.value}

    def _simulation(self, idx: int):
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

    def _run_simulation(self, idx, command):
        if command == 0:
            self.database.tree_clear(idx)

        start_time = time()
        while (time() - start_time) < self.calculation_time:
            self._simulation(idx)

    def _process_paths(self, idx, done, last_value, simulation_path):
        pass

    def run(self):
        idx = self.idx

        self.pred_time = []
        self.run_time = []
        self.num_nodes = 0

        while True:
            self.input_queue.wait()
            self.input_queue.clear()
            command = self.input_param.value

            if command == -1:
                break

            self._run_simulation(idx, command)
            
            self.output_num_nodes.value = self.num_nodes
            self.output_pred_time.value = np.mean(self.pred_time)
            self.output_run_time.value = np.mean(self.run_time)
            self.output_queue.set()

            self.pred_time.clear()
            self.run_time.clear()
            self.num_nodes = 0


class SimulationProcessManager:
    def __init__(self, num_workers: int, env, network_manager: NetworkManager, database: DataProcess, mcts_config: dict,
                 *, worker_class=SimulationProcess, **worker_args):
        self.num_workers = num_workers
        self.env = env
        self.state_shape = self.env.state_shape
        self.network_manager = network_manager
        self.database = database
        self.mcts_config = mcts_config

        self.state_buffer_base = Array(ctypes.c_int64, int(np.prod(env.state_shape)), lock=False)
        self.starting_state = np.ctypeslib.as_array(self.state_buffer_base)
        self.starting_state = self.starting_state.reshape(env.state_shape)

        self.workers = [worker_class(idx, env, network_manager, database,
                                          self.state_buffer_base, self.state_shape, mcts_config, **worker_args)
                        for idx in range(num_workers)]

    def start(self) -> None:
        """ Start all workers. """
        for worker in self.workers:
            worker.start()

    def shutdown(self) -> None:
        """ Kill workers without mercy. """
        print("Killing Worker:", end="")
        for i, worker in enumerate(self.workers):
            if worker.pid:
                print(" {}".format(i), end="")
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

