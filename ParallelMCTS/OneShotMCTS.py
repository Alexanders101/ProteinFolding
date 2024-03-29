import numpy as np
import tensorflow as tf

from typing import Callable, Tuple
from ParallelMCTS.OptimalSimulationProcess import OptimalSimulationProcessManager
from ParallelMCTS.ParallelMCTS import ParallelMCTS
from ParallelMCTS.SinglePlayerEnvironment import SinglePlayerEnvironment


class OneShotMCTS(ParallelMCTS):
    def __init__(self, env: SinglePlayerEnvironment, make_model: Callable[[], tf.keras.Model],
                 num_parallel: int = 1, num_workers: int = 2, num_networks: int = 4,
                 session_config: tf.ConfigProto = None, *,
                 max_to_keep=5, network_options: dict = {}, database_options: dict = {}, **kwargs):
        super(OneShotMCTS, self).__init__(env, make_model, num_parallel, num_workers, num_networks, session_config,
                                          simulation_manager=OptimalSimulationProcessManager,
                                          simulation_options={'max_to_keep': max_to_keep},
                                          network_options=network_options,
                                          database_options=database_options,
                                          **kwargs)

    def select_moves(self, worker_idx: int, state: np.ndarray, state_depth: int = 0) -> np.ndarray:
        raise NotImplementedError(
            "OneshotMCTS does not select individual moves. Use 'play' method to gather best paths.")

    def select_moves_async(self, worker_idx: int, state: np.ndarray, state_depth: int = 0):
        raise NotImplementedError(
            "OneshotMCTS does not select individual moves. Use 'play' method to gather best paths.")

    def select_moves_result(self, worker_idx: int):
        raise NotImplementedError(
            "OneshotMCTS does not select individual moves. Use 'play' method to gather best paths.")

    def play(self, worker_idx: int, start_state: np.ndarray, clear: bool = True) -> Tuple[float, np.ndarray]:
        if clear:
            self.clear(worker_idx)

        worker = self.workers[worker_idx]

        # Start simulation on new state
        worker.set_start_state(start_state)
        worker.simulation(clear_tree=True)

        # Block and wait for results
        results = worker.results()

        if self.verbose >= 2:
            print("{} Nodes per second".format(np.sum(results['total_nodes']) / self.calculation_time))

        return worker.get_optimal_solutions()

    def fit_epoch_single(self, num_games: int = 1, worker_idx: int = 0):
        raise NotImplementedError("Training Not implemented for OneshotMCTS yet.")

    def fit_epoch_multi(self, num_games_per_worker: int = 1):
        raise NotImplementedError("Training Not implemented for OneshotMCTS yet.")
