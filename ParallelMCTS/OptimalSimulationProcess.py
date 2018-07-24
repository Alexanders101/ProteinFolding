import numpy as np
from ParallelMCTS.SimulationProcess import SimulationProcess, SimulationProcessManager
from ParallelMCTS.NetworkManager import NetworkManager
from ParallelMCTS.DataProcess import DataProcess

from multiprocessing import Array, Value
from ctypes import c_int64, c_float


class OptimalSimulationProcessManager(SimulationProcessManager):
    def __init__(self, num_workers: int, env, network_manager: NetworkManager, database: DataProcess, mcts_config: dict,
                 max_to_keep: int = 10):
        super(OptimalSimulationProcessManager, self).__init__(num_workers, env, network_manager, database, mcts_config,
                                                              worker_class=OptimalSimulationProcess,
                                                              max_to_keep=max_to_keep)
        self.max_to_keep = max_to_keep

    def get_optimal_solutions(self):
        self.results()

        values = np.empty(self.num_workers, dtype=np.float32)
        paths = np.empty((self.num_workers, self.max_to_keep, self.env.max_length), dtype=np.int64)

        for i, worker in enumerate(self.workers):
            values[i] = worker.best_value.value
            paths[i, :, :] = worker.best_paths[:]

        best_value = values.max()
        best_value_idx = np.argwhere(values == best_value)
        paths = paths[best_value_idx].reshape((-1, self.env.max_length))

        return best_value, np.unique(paths, axis=0)


class OptimalSimulationProcess(SimulationProcess):
    def __init__(self, *args, max_to_keep=5, **kwargs):
        super(OptimalSimulationProcess, self).__init__(*args, **kwargs)
        self.max_to_keep = max_to_keep

        self.__best_paths_base = Array(c_int64, max_to_keep * self.env.max_length, lock=False)
        self.best_paths = np.frombuffer(self.__best_paths_base, dtype=np.int64, count=max_to_keep * self.env.max_length)
        self.best_paths = self.best_paths.reshape((max_to_keep, self.env.max_length))

        self.best_value = Value(c_float, lock=False)

    def _run_simulation(self, idx, command):
        self.best_value.value = -np.float32(np.inf)
        self.num_paths = 0
        self.best_paths.fill(-1)

        super(OptimalSimulationProcess, self)._run_simulation(idx, command)

    def _process_paths(self, idx, done, last_value, simulation_path):
        if not done or last_value < self.best_value.value:
            return

        # Extract the full path to solution from the simulation
        path_size = len(simulation_path)
        path = np.fromiter(map(lambda x: x[1], simulation_path), dtype=np.int64, count=path_size)

        # If we have found a new path with the best value, add it to the store if there is space
        if last_value == self.best_value.value and self.num_paths < self.max_to_keep:
            self.best_paths[self.num_paths, :path_size] = path
            self.num_paths += 1

        # If we have found a better bath, store it and update best value
        if last_value > self.best_value.value:
            self.best_paths[:self.num_paths] = -1
            self.best_paths[0, :path_size] = path
            self.num_paths = 1
            self.best_value.value = last_value
