from multiprocessing import Process, Queue, Array
from NetworkProcess import NetworkProcess
from DataProcess import DataProcess
import numpy as np
import ctypes
from time import time


class SimulationProcessManager:
    def __init__(self, num_workers: int, env, network_process: NetworkProcess, database: DataProcess, mcts_config: dict):
        self.num_workers = num_workers
        self.env = env
        self.state_shape = self.env.state_shape
        self.network_process = network_process
        self.database = database
        self.mcts_config = mcts_config

        self.state_buffer_base = Array(ctypes.c_int64, int(np.prod(env.state_shape)), lock=False)
        self.starting_state = np.ctypeslib.as_array(self.state_buffer_base)
        self.starting_state = self.starting_state.reshape(env.state_shape)

        self.workers = [SimulationProcess(idx, env, network_process, database,
                                          self.state_buffer_base, self.state_shape, mcts_config)
                        for idx in range(num_workers)]

    def start(self):
        for worker in self.workers:
            worker.start()

    def shutdown(self):
        for worker in self.workers:
            worker.shutdown()

    def set_start_state(self, state):
        self.starting_state[:] = state.copy()

    def simulation(self, clear_tree: bool = True):
        for worker in self.workers:
            worker.simulation(clear_tree)

    def results(self):
        res = [worker.result() for worker in self.workers]
        return {k: [dic[k] for dic in res] for k in res[0]}

class SimulationProcess(Process):
    def __init__(self, idx: int, env, network_process: NetworkProcess, database: DataProcess,
                 state_buffer_base: Array, state_shape: tuple, mcts_config: dict):
        super(SimulationProcess, self).__init__()

        self.idx = idx
        self.env = env
        self.database = database
        self.network_process = network_process

        self.starting_state = np.ctypeslib.as_array(state_buffer_base)
        self.starting_state = self.starting_state.reshape(state_shape)

        self.input_queue = Queue(1)
        self.output_queue = Queue()

        self.calculation_time = mcts_config['calculation_time']
        self.C = mcts_config['C']
        self.epsilon = mcts_config['epsilon']
        self.alpha = mcts_config['alpha']
        self.virtual_loss = mcts_config['virtual_loss']
        self.verbose = mcts_config['verbose']

    def shutdown(self):
        self.input_queue.put(-1)

    def simulation(self, clear_tree: bool = True):
        if clear_tree:
            self.input_queue.put(0)
        else:
            self.input_queue.put(1)

    def result(self):
        res = self.output_queue.get()
        return {'num_nodes': res[0], 'mean_pred_time': res[1], 'mean_run_time': res[2]}

    def __simulation(self, idx: int, tree: set):
        simulation_path = []
        state = self.starting_state.copy()
        state_hash = self.env.hash(self.starting_state)

        # #####
        # Create the necessary data for the root node
        # #############################################
        next_states = np.concatenate((self.env.next_state_multi(state, self.env.moves), np.expand_dims(state, 0)))
        policy, values = self.network_process._predict_unsafe(idx, next_states)
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
            N, _, Q, V = self.database.get(idx, state_hash)
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
                break

            next_states = np.concatenate((self.env.next_state_multi(state, self.env.moves), np.expand_dims(state, 0)))
            ###########################################################
            self.run_time.append(time() - t0)
            t0 = time()
            ###########################################################
            policy, values = self.network_process._predict_unsafe(idx, next_states)
            ###########################################################
            self.pred_time.append(time() - t0)
            t0 = time()
            ###########################################################
            policy = policy[-1]
            values = values[:-1, 0]

        # Initialize new node
        self.database.add(state_hash)
        tree.add(state_hash)

        for hash, action in simulation_path:
            self.database.backup(hash, action, last_value)

    def run(self):
        idx = self.idx
        tree = set()

        self.pred_time = []
        self.run_time = []
        self.num_nodes = 0

        while True:
            command = self.input_queue.get()
            if command == -1:
                break

            if command == 0:
                tree.clear()

            start_time = time()
            while (time() - start_time) < self.calculation_time:
                self.__simulation(idx, tree)

            self.output_queue.put((self.num_nodes, np.mean(self.pred_time), np.mean(self.run_time)))
            self.pred_time.clear()
            self.run_time.clear()
            self.num_nodes = 0

