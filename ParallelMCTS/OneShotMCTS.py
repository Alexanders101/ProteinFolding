from ParallelMCTS.ParallelMCTS import ParallelMCTS
from ParallelMCTS.OptimalSimulationProcess import OptimalSimulationProcessManager
from ParallelMCTS.SinglePlayerEnvironment import SinglePlayerEnvironment
import tensorflow as tf
import numpy as np
from typing import Callable
from time import time


class OneShotMCTS(ParallelMCTS):
    def __init__(self, env: SinglePlayerEnvironment, make_model: Callable[[], tf.keras.Model],
                 num_threads: int =2, num_networks: int = 4, session_config: tf.ConfigProto = None, *,
                 max_to_keep=5, network_options: dict = {}, database_options: dict = {}, **kwargs):
        super(OneShotMCTS, self).__init__(env, make_model, num_threads, num_networks, session_config,
                                          simulation_manager=OptimalSimulationProcessManager,
                                          simulation_options={'max_to_keep': max_to_keep},
                                          network_options=network_options,
                                          database_options=database_options,
                                          **kwargs)

    def select_moves(self, state: np.ndarray):
        raise NotImplementedError("OneshotMCTS does not select individual moves. Use play method instead.")

    def play(self, start_state: np.ndarray, clear: bool = True):
        if clear:
            self.clear()

        episode_start_time = time()

        # Start simulation on new state
        self.workers.set_start_state(start_state)
        self.workers.simulation(clear_tree=True)

        # Block and wait for results
        results = self.workers.results()

        if self.verbose >= 2:
            print("{} Nodes per second".format(np.sum(results['num_nodes']) / (time() - episode_start_time)))
            print("Avg RunTime: {}".format(np.mean(results['mean_run_time'])))
            print("Avg PredTime: {}".format(np.mean(results['mean_pred_time'])))

        return self.workers.get_optimal_solutions()

