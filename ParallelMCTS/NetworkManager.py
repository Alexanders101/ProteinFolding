from multiprocessing import Process, Queue, Array, Event
from ParallelMCTS.DistributedNetworkProcess import DistributedNetworkProcess, DistributedTrainingProcess
from ParallelMCTS.SinglePlayerEnvironment import np2c
import tensorflow as tf
from tensorflow import keras
import ctypes
import numpy as np
from numba import jit
from typing import Callable, Tuple
import os
import signal
from time import sleep


@jit("void(int64[::1], int64[::1], int64[::1])", nopython=True)
def counting_unique_sort(arr: np.ndarray, buckets: np.ndarray, output: np.ndarray):
    """ This function performs an integer counting sort.

    Runtime is O(k) where k is the maximum possible value of the input.

    Parameters
    ----------
    arr : np.ndarray[int, 1]
        Input array. The size is (n).
    buckets : np.ndarray[int, 1]
        Pre-allocated array of buckets, these must all be zero. The size must be (k).
    output : np.ndarray[int, 1]
        Pre-allocated array to place results. Shape must match (n).
    """
    n = arr.shape[0]
    k = buckets.shape[0]
    for i in range(n):
        buckets[arr[i]] = 1

    index = 0
    for i in range(k):
        if buckets[i] > 0:
            buckets[i] = 0
            output[index] = i
            index += 1


class NetworkManager(Process):
    def __init__(self, make_network: Callable[[], keras.Model], state_shape: Tuple[int, ...], state_type: type,
                 num_moves: int, num_states: int, num_workers: int = 1, num_networks: int = 1,
                 session_config: tf.ConfigProto = None, *, batch_size: int = None, train_buffer_size: int = 64,
                 start_port: int = 2222, num_ps: float = None, **kwargs):
        """ Class for managing distributed Tensorflow models and predict / train asynchronously.

        Parameters
        ----------
        make_network : () -> Keras.Model
            Function defining how to construct your network.
        state_shape : (int, ...)
            Shape of a single input state.
        num_moves : int
            Number of moves possible.
        num_states : int
            How many states you plan to feed in at any given time.
        num_workers : int
            How many asynchronous workers will utilize this class for prediction.
        num_networks : int
            Number of network instances to create.
        session_config : tf.ConfigProto
            Tensorflow session config object.

        batch_size : int -- Named Only, default = None
            Batch size network uses for prediction. This should be a multiple of num_states.
            If None, even batches are calculated.
        train_buffer_size : int -- Named Only, default = 64
            Batch size for training. This will create train buffers for that batch size, do not put more
            data than that into a buffer.
        start_port : int -- Named Only, default = 2222
            Starting port for the tensorflow servers.
        num_ps : float -- Named Only, default = None
            Sets the number of parameter servers.
            If None - Sets to number of GPUs
            If 0 < num_ps < 1 - Treat num_ps as a ratio of the number of workers.
            If num_ps >= 1 - Treat num_ps as the number of servers.
        """
        super(NetworkManager, self).__init__()
        self.make_network = make_network
        self.session_config = session_config

        if batch_size is None:
            batch_size = (num_workers * num_states) // num_networks

        self.state_shape = state_shape
        self.num_moves = num_moves
        self.num_states = num_states
        self.batch_size = batch_size
        self.train_buffer_size = train_buffer_size
        self.num_samples = batch_size // num_states

        self.num_workers = num_workers
        self.num_networks = num_networks

        # ############################################################################################
        # Communication Objects
        # ############################################################################################

        # ############################################################################################
        # WORKER INPUT OUTPUT
        # Workers will put their worker ID onto this queue once their input is on the input buffer.
        # Each worker will only put one request on here at a time, so maxsize is safe.
        self.input_queue = Queue(maxsize=num_workers)

        # Networks will signal to workers once their outputs are in the buffers.
        # Each worker gets their own ready event, the must reset it when predicting
        # and wait for the network to Set it True.
        self.output_ready = [Event() for _ in range(num_workers)]

        # ############################################################################################
        # NETWORK INPUT OUTPUT
        # Manager will put how many samples the network should process from index buffer.
        self.network_input_queue = [Queue(1) for _ in range(num_networks)]

        # Networks will signal to manager when they are ready for a new task.
        # Manager must clear these once input is ready to predict and wait for a network to be ready before
        # Placing new data onto input queue.
        self.network_ready = [Event() for _ in range(num_networks)]

        # ############################################################################################
        # TRAINING INPUT OUTPUT
        # Master signals to training worker how much data to train on.
        self.training_input_queue = Queue(maxsize=1)

        # Training Network will signal to master when training has finished.
        self.training_ready = Event()
        self.training_ready.set()

        # ############################################################################################
        # Buffers
        # ############################################################################################
        state_type = np2c(state_type)
        c_int64 = ctypes.c_int64
        c_float = ctypes.c_float

        # INPUT BUFFER - Workers place input states onto here.
        input_buffer_shape = (num_workers, num_states, *state_shape)
        self.__input_buffer_base = Array(state_type, int(np.prod(input_buffer_shape)), lock=False)
        self.input_buffer = np.ctypeslib.as_array(self.__input_buffer_base).reshape(input_buffer_shape)

        # INDEX BUFFERS - Manager places the indices of network inputs onto here.
        index_buffers_shape = (num_networks, self.num_samples)
        self.__index_buffers_base = Array(c_int64, int(np.prod(index_buffers_shape)), lock=False)
        self.index_buffers = np.ctypeslib.as_array(self.__index_buffers_base).reshape(index_buffers_shape)

        # POLICY BUFFER - Networks will place results here and worker can read from here.
        policy_buffer_shape = (num_workers, num_states, num_moves)
        self.__policy_buffer_base = Array(c_float, int(np.prod(policy_buffer_shape)), lock=False)
        self.policy_buffer = np.ctypeslib.as_array(self.__policy_buffer_base).reshape(policy_buffer_shape)

        # VALUE BUFFER - Networks will place results here and workers can read from here.
        value_buffer_shape = (num_workers, num_states, 1)
        self.__value_buffer_base = Array(c_float, int(np.prod(value_buffer_shape)), lock=False)
        self.value_buffer = np.ctypeslib.as_array(self.__value_buffer_base).reshape(value_buffer_shape)

        # TRAINING_BUFFER - Master places states for training here.
        training_buffer_shape = (train_buffer_size, *state_shape)
        self.__training_buffer_base = Array(state_type, int(np.prod(training_buffer_shape)), lock=False)
        self.training_buffer = np.ctypeslib.as_array(self.__training_buffer_base).reshape(training_buffer_shape)

        # POLICY TARGET BUFFER - Master places policy targets for training here.
        policy_target_buffer_shape = (train_buffer_size, num_moves)
        self.__policy_target_buffer_base = Array(c_float, int(np.prod(policy_target_buffer_shape)), lock=False)
        self.policy_target_buffer = np.ctypeslib.as_array(self.__policy_target_buffer_base).reshape(
            policy_target_buffer_shape)

        # VALUE TARGET BUFFER - Master places value targets for training here.
        value_target_buffer_shape = (train_buffer_size, 1)
        self.__value_target_buffer_base = Array(c_float, int(np.prod(value_target_buffer_shape)), lock=False)
        self.value_target_buffer = np.ctypeslib.as_array(self.__value_target_buffer_base).reshape(
            value_target_buffer_shape)

        # ############################################################################################
        #  Network Creation
        # ############################################################################################
        # Calculate number of servers.
        # Default to number of GPUs available.
        if num_ps is None:
            try:
                visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
                visible_devices = visible_devices.split(',')
                num_ps = len(visible_devices)
            except KeyError:
                num_ps = 1

        # If ratio between 0 and 1, calculate number of workers
        elif num_ps < 1:
            num_ps = int(num_ps * num_networks)

        # Otherwise treat as raw value
        else:
            num_ps = int(num_ps)

        num_ps = max(num_ps, 1)

        # Cluster config uses only localhost to maintain multiple tensorflow instances.
        cluster_spec = {"worker": ["localhost:{}".format(start_port+i) for i in range(num_networks+1)],
                        "ps": ["localhost:{}".format(start_port+num_networks+i+1) for i in range(num_ps)]}
        cluster_spec = tf.train.ClusterSpec(cluster_spec)
        self.cluster_spec = cluster_spec

        # Create Network Workers
        self.networks = []
        for i in range(num_networks):
            network = DistributedNetworkProcess(make_network, session_config,
                                                task_index=i+1,
                                                parameter_server=False,
                                                cluster_spec=cluster_spec,
                                                input_queue=self.network_input_queue[i],
                                                ready_event=self.network_ready[i],
                                                output_ready=self.output_ready,
                                                input_buffer=self.input_buffer,
                                                index_buffer=self.index_buffers[i],
                                                policy_buffer=self.policy_buffer,
                                                value_buffer=self.value_buffer,
                                                **kwargs)
            self.networks.append(network)

        # Create Parameter Servers
        self.parameter_servers = []
        for i in range(num_ps):
            # noinspection PyTypeChecker
            ps = DistributedNetworkProcess(make_network, session_config,
                                           task_index=i,
                                           parameter_server=True,
                                           cluster_spec=cluster_spec,
                                           input_queue=None,
                                           ready_event=None,
                                           output_ready=None,
                                           input_buffer=None,
                                           index_buffer=None,
                                           policy_buffer=None,
                                           value_buffer=None,
                                           **kwargs)
            self.parameter_servers.append(ps)

        # Create Training Network.
        self.training_network = DistributedTrainingProcess(make_network, session_config,
                                                           task_index=0,
                                                           cluster_spec=cluster_spec,
                                                           input_queue=self.training_input_queue,
                                                           ready_event=self.training_ready,
                                                           training_buffer=self.training_buffer,
                                                           policy_target_buffer=self.policy_target_buffer,
                                                           value_target_buffer=self.value_target_buffer,
                                                           **kwargs)

    def __str__(self):
        cluster_spec = self.cluster_spec.as_dict()

        out = ["=" * 60]
        out.append(super(NetworkManager, self).__str__())
        out.append("-"*60)
        out.append("Number of Prediction Networks: {}".format(self.num_networks))
        out.append("Number of Parameter Servers: {}".format(len(cluster_spec['ps'])))
        out.append("")
        out.append("Cluster Specification: ")
        out.append("-" * 60)
        out.append("    Prediction Network: {}".format(cluster_spec['worker'][0]))
        out.append("-" * 60)
        for i, worker in enumerate(cluster_spec['worker'][1:]):
            out.append("    Worker Network {}: {}".format(i, worker))
        out.append("-"*60)
        for i, ps in enumerate(cluster_spec['ps']):
            out.append("    Parameter Server {}: {}".format(i, ps))
        out.append("-" * 60)

        return "\n".join(out)

    def wait_until_all_ready(self) -> None:
        """ Blocks until all networks have initialized. """
        for ready in self.network_ready:
            ready.wait()
        self.training_ready.wait()

    def ready_workers(self) -> [int]:
        """ Get all the prediction networks that are ready to predict.

        Returns
        -------
        [int]
            List of networks by network id.
        """
        result = []
        for i, ready in enumerate(self.network_ready):
            if ready.is_set():
                result.append(i)
        return result

    def predict(self, idx: int, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Schedule a predict request. This will block until the results are ready.

        Parameters
        ----------
        idx : int
            Index of worker.
        states : np.ndarray[int, [None, ...]]
            A single batch of states.

        Returns
        -------
        policy : np.ndarray[float, [None, num_moves]]
        value : np.ndarray[float, [None, 1]]
        """
        return self._predict_unsafe(idx, states)

    def predict_single(self, idx: int, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """ Schedule a predict request for a single state. This will block until the results are ready.

        This is identical to predict in terms of how it works. This is just a helper function to avoid having to
        batch single states all the time.

        Parameters
        ----------
        idx : int
            Index of worker.
        state : np.ndarray[int, [None, ...]]
            A single, non-batched, state

        Returns
        -------
        policy : np.ndarray[float, [num_moves]]
        value : float
        """
        self.input_buffer[idx, 0] = state[:]
        self.output_ready[idx].clear()
        self.input_queue.put_nowait(idx)

        self.output_ready[idx].wait()

        policy = self.policy_buffer[idx, 0]
        policy = np.exp(policy)
        policy /= np.sum(policy)

        return policy, self.value_buffer[idx, 0, 0]

    def _predict_unsafe(self, idx, states):
        num_samples = states.shape[0]

        self.input_buffer[idx, :num_samples] = states[:]
        self.output_ready[idx].clear()
        self.input_queue.put_nowait(idx)

        self.output_ready[idx].wait()

        policy = self.policy_buffer[idx, :num_samples]
        policy = np.exp(policy)
        policy /= np.sum(policy, 1)

        return policy, self.value_buffer[idx, :num_samples]

    def fit(self, states: np.ndarray, policy_targets: np.ndarray, value_targets: np.ndarray) -> None:
        """ Schedules a training request.

        This will block if there is training in progress. It will continue to block until the training network is free.

        Parameters
        ----------
        states : np.ndarray[int, [None, ...]]
            A batch of states.
        policy_targets : np.ndarray[float, [None, num_moves]]
            Targets for the policy network.
        value_targets : np.ndarray[float, [None, 1]]
            Targets for the value network.
        """
        self.training_ready.wait()

        num_samples = states.shape[0]

        self.training_buffer[:num_samples] = states
        self.policy_target_buffer[:num_samples] = policy_targets
        self.value_target_buffer[:num_samples] = value_targets

        self.training_ready.clear()
        self.training_input_queue.put((0, num_samples))

    def save_weights(self, weight_file: str) -> None:
        """ Schedule a save_weights request. This will save the current weights of the network to a Keras hdf5 file.

        Parameters
        ----------
        weight_file : str
            File to store weights to.
        """
        self.training_ready.wait()

        self.training_ready.clear()
        self.training_input_queue.put((1, weight_file))

    def __enter__(self):
        self.start()
        self.wait_until_all_ready()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def shutdown(self) -> None:
        """ Kill all networks violently. """
        for i, network in enumerate(self.networks):
            if network.pid:
                print("Killing Network Worker: {}".format(i))
                os.kill(network.pid, signal.SIGKILL)

        if self.training_network.pid:
            print("Killing Training Network")
            os.kill(self.training_network.pid, signal.SIGKILL)

        for i, ps in enumerate(self.parameter_servers):
            if ps.pid:
                print("Killing Parameter Server: {}".format(i))
                os.kill(ps.pid, signal.SIGKILL)

        print("Shutting Down Manager")
        self.input_queue.put(-1)

    def start(self, wait_time: float = 3):
        for i, network in enumerate(self.networks):
            print("Starting Prediction Network {}".format(i))
            network.start()

        for i, ps in enumerate(self.parameter_servers):
            print("Starting Parameter Server {}".format(i))
            ps.start()

        sleep(wait_time)
        print("Starting Training Network.")
        self.training_network.start()

        print("Starting Network Manager.")
        super(NetworkManager, self).start()

    def run(self):
        # Store Variables locally for faster access
        max_samples = self.num_samples
        num_networks = self.num_networks

        input_queue = self.input_queue
        network_ready = self.network_ready
        network_input_queue = self.network_input_queue
        index_buffers = self.index_buffers

        # Iterate through the networks as a round robin.
        current_network = 0

        # This buffer stores the worker indices when gathering data to predict
        ids = np.zeros(max_samples, dtype=np.int64)

        # This is a constant buffer for performing counting sort on indices.
        buckets = np.zeros(self.num_workers, dtype=np.int64)

        while True:
            # Wait for next network to be ready to predict.
            network_ready[current_network].wait()

            # Get the first index from the queue and kill process if receive -1.
            ids[0] = input_queue.get()
            if ids[0] < 0:
                break

            # Gather as many requests as possible.
            size = 1
            while size < max_samples and not input_queue.empty():
                ids[size] = input_queue.get()
                size += 1

            # Sort request indices for much faster numpy indexing.
            # Store the sorted indices directly on index buffer.
            counting_unique_sort(ids[:size], buckets, output=index_buffers[current_network, :])

            # Ready network for prediction and place the number of inputs the network must process.
            network_ready[current_network].clear()
            network_input_queue[current_network].put(size)

            current_network = (current_network + 1) % num_networks
