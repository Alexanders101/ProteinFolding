from multiprocessing import Process, Queue, Array, Event
import tensorflow as tf
from tensorflow import keras
import ctypes
import numpy as np
from numba import jit
from typing import Callable, Tuple
import os
import signal


@jit("void(int16[::1], int16[::1], int16[::1])", nopython=True)
def counting_unique_sort(arr, buckets, output):
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
    def __init__(self, make_network: Callable[[], keras.Model], state_shape: Tuple[int, ...],
                 num_moves: int, num_states: int, batch_size: int, num_workers: int, num_networks: int,
                 session_config: tf.ConfigProto = None):
        """ Class for managing distributed Tensorflow models and predict / train asynchronously.

        Parameters
        ----------
        make_network : () -> keras.Model
            Function defining how to construct your network.
        state_shape : (int, ...)
            Shape of a single input state.
        num_moves : int
            Number of moves possible.
        num_states : int
            How many states you plan to feed in at any given time.
        batch_size : int
            Batch size network uses for prediction / training. This should be a multiple of num_states.
        num_workers : int
            How many asynchronous workers will utilize this class for prediction.
        num_networks : int
            Number of network instances to create.
        session_config : tf.ConfigProto
            Tensorflow session config object.
        """
        super(NetworkManager, self).__init__()
        self.make_network = make_network
        self.session_config = session_config

        self.state_shape = state_shape
        self.num_moves = num_moves
        self.num_states = num_states
        self.batch_size = batch_size
        self.num_samples = batch_size // num_states

        self.num_workers = num_workers
        self.num_networks = num_networks

        # Workers will put their worker ID onto this queue once their input is on the input buffer.
        self.input_queue = Queue(maxsize=num_workers)

        # Networks will signal to workers once their outputs are in the buffers.
        self.output_queue = [Event() for _ in range(num_workers)]

        # Manager will put how many samples the network should process from index buffer.
        self.network_input_queue = [Queue(1) for _ in range(num_networks)]

        # Networks will signal to manager when they are ready for a new task.
        self.network_ready = [Event() for _ in range(num_networks)]

        # Type quickies
        c_int64 = ctypes.c_int64
        c_int16 = ctypes.c_int16
        c_float = ctypes.c_float

        # INPUT BUFFER - Workers place input states onto here.
        input_buffer_shape = (num_workers, num_states, *state_shape)
        self.__input_buffer_base = Array(c_int64, int(np.prod(input_buffer_shape)), lock=False)
        self.input_buffer = np.ctypeslib.as_array(self.__input_buffer_base).reshape(input_buffer_shape)

        # INDEX BUFFERS - Manager places the indices of network inputs onto here.
        index_buffers_shape = (num_networks, self.num_samples)
        self.__index_buffers_base = Array(c_int16, int(np.prod(index_buffers_shape)), lock=False)
        self.index_buffers = np.ctypeslib.as_array(self.__index_buffers_base).reshape(index_buffers_shape)

        # POLICY BUFFER - Networks will place results here and worker can read from here.
        policy_buffer_shape = (num_workers, num_states, num_moves)
        self.__policy_buffer_base = Array(c_float, int(np.prod(policy_buffer_shape)), lock=False)
        self.policy_buffer = np.ctypeslib.as_array(self.__policy_buffer_base).reshape(policy_buffer_shape)

        # VALUE BUFFER - Networks will place results here and workers can read from here.
        value_buffer_shape = (num_workers, num_states, 1)
        self.__value_buffer_base = Array(c_float, int(np.prod(value_buffer_shape)), lock=False)
        self.value_buffer = np.ctypeslib.as_array(self.__value_buffer_base).reshape(value_buffer_shape)



        # Create Network Subprocesses
        PS_RATIO = 0.5
        START_PORT = 2222
        num_ps = int(PS_RATIO * num_networks)
        num_ps = max(num_ps, 1)

        cluster_spec = {"worker": ["localhost:{}".format(START_PORT+i) for i in range(num_networks)],
                        "ps": ["localhost:{}".format(START_PORT+num_networks+i) for i in range(num_ps)]}
        cluster_spec = tf.train.ClusterSpec(cluster_spec)

        self.networks = []
        for i in range(num_networks):
            network = DistributedNetworkProcess(make_network, session_config,
                                                task_index=i, parameter_server=False, cluster_spec=cluster_spec,
                                                input_queue=self.network_input_queue[i],
                                                ready_event=self.network_ready[i],
                                                output_queue=self.output_queue, input_buffer=self.input_buffer,
                                                index_buffer=self.index_buffers[i],
                                                policy_buffer=self.policy_buffer, value_buffer=self.value_buffer)
            network.start()
            self.networks.append(network)

        self.parameter_servers = []
        for i in range(num_ps):
            ps = DistributedNetworkProcess(make_network, session_config,
                                           task_index=i, parameter_server=True, cluster_spec=cluster_spec,
                                           input_queue=None, ready_event=None, output_queue=None, input_buffer=None,
                                           index_buffer=None, policy_buffer=None, value_buffer=None)
            ps.start()
            self.parameter_servers.append(ps)

    def __del__(self):
        self.shutdown()

    def wait_until_all_ready(self):
        for ready in self.network_ready:
            ready.wait()

    def ready_workers(self):
        result = []
        for i, ready in enumerate(self.network_ready):
            if ready.is_set():
                result.append(i)
        return result

    def predict(self, idx, states):
        self.input_buffer[idx, :] = states[:]
        self.output_queue[idx].clear()
        self.input_queue.put_nowait(idx)

        self.output_queue[idx].wait()
        return self.policy_buffer[idx], self.value_buffer[idx]

    def shutdown(self):
        for i, network in enumerate(self.networks):
            if network.pid:
                print("Killing Network Worker: {}".format(i))
                os.kill(network.pid, signal.SIGKILL)

        for i, ps in enumerate(self.parameter_servers):
            if ps.pid:
                print("Killing Parameter Server: {}".format(i))
                os.kill(ps.pid, signal.SIGKILL)

        print("Shutting Down Manager")
        self.input_queue.put(-1)


    def run(self):
        # Store Variables locally for fast access
        max_samples = self.num_samples
        num_networks = self.num_networks

        input_queue = self.input_queue
        network_ready = self.network_ready
        network_input_queue = self.network_input_queue
        index_buffers = self.index_buffers

        current_network = 0

        # data buffers
        ids = np.zeros(max_samples, dtype=np.int16)
        buckets = np.zeros(self.num_workers, dtype=np.int16)

        while True:
            network_ready[current_network].wait()
            ids[0] = input_queue.get()
            if ids[0] < 0:
                break

            size = 1
            while size < max_samples and not input_queue.empty():
                ids[size] = input_queue.get()
                size += 1

            counting_unique_sort(ids[:size], buckets, output=index_buffers[current_network, :])

            network_ready[current_network].clear()
            network_input_queue[current_network].put(size)

            current_network = (current_network + 1) % num_networks

class DistributedNetworkConfig:
    def __init__(self, learning_rate=0.01, policy_weight=1.0, tensorboard_log=False, **kwargs):
        self.learning_rate = learning_rate
        self.policy_weight = policy_weight
        self.tensorboard_log = tensorboard_log

class DistributedNetworkProcess(Process):
    def __init__(self, make_network: Callable[[], keras.Model], session_config: tf.ConfigProto,
                 task_index: int, parameter_server: bool, cluster_spec: tf.train.ClusterSpec,
                 input_queue: Queue, ready_event: Event, output_queue: [Queue],
                 input_buffer: np.ndarray, index_buffer: np.ndarray,
                 policy_buffer: np.ndarray, value_buffer: np.ndarray, **kwargs):
        super(DistributedNetworkProcess, self).__init__()

        self.make_network = make_network
        self.session_config = session_config
        self.network_config = DistributedNetworkConfig(**kwargs)

        self.task_index = task_index
        self.cluster_spec = cluster_spec
        self.parameter_server = parameter_server

        self.input_queue = input_queue
        self.ready_event = ready_event
        self.output_queue = output_queue

        self.input_buffer = input_buffer
        self.index_buffer = index_buffer
        self.policy_buffer = policy_buffer
        self.value_buffer = value_buffer

    def __initialize_network(self):
        keras.backend.manual_variable_initialization(True)

        device = "/job:worker/task:{}".format(self.task_index)
        with tf.device(tf.train.replica_device_setter(worker_device=device, cluster=self.cluster_spec)):
            self.global_step = tf.train.get_or_create_global_step()
            self.training_phase = keras.backend.learning_phase()

            with tf.name_scope("Targets"):
                num_moves = self.policy_buffer.shape[-1]
                self.policy_target = tf.placeholder(tf.float32, shape=(None, num_moves), name="PolicyTargets")
                self.value_target = tf.placeholder(tf.float32, shape=(None, 1), name="ValueTargets")

            with tf.name_scope("Network"):
                self.model = self.make_network()

            self.x = self.model.input
            self.policy, self.value = self.model.output

            with tf.name_scope("Loss"):
                with tf.name_scope("ValueLoss"):
                    value_loss = tf.losses.mean_squared_error(self.value_target, self.value)
                    self.value_loss = tf.reduce_mean(value_loss)

                with tf.name_scope("PolicyLoss"):
                    policy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy_target,
                                                                             logits=self.policy)
                    self.policy_loss = policy_loss

                with tf.name_scope("TotalLoss"):
                    policy_weight = self.network_config.policy_weight
                    policy_weight = policy_weight / (policy_weight + 1)
                    value_weight = 1 - policy_weight
                    self.total_loss = (policy_weight * self.policy_loss) + (value_weight * self.value_loss)

            with tf.name_scope("Optimizer"):
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.network_config.learning_rate)
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)
                self.train_op = tf.group(self.train_op, *self.model.updates)

            if self.network_config.tensorboard_log:
                for layer in self.model.layers:
                    with tf.name_scope(layer.name):
                        for weight in layer.weights:
                            with tf.name_scope(weight.name.split("/")[-1].split(":")[0]):
                                tf.summary.histogram('histogram', weight)

    def run(self):
        # Create and start a server for the local task.
        job_name = "ps" if self.parameter_server else "worker"
        server = tf.train.Server(self.cluster_spec, job_name=job_name, task_index=self.task_index,
                                 config=self.session_config)

        # Parameter Server chills here
        if self.parameter_server:
            server.join()
            return

        # The workers continue
        self.__initialize_network()

        # Add hooks if necessary
        hooks = None
        chief_only_hooks = None

        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=self.task_index == 0,
                                               hooks=hooks, chief_only_hooks=chief_only_hooks,
                                               config=self.session_config,
                                               save_checkpoint_secs=None, save_checkpoint_steps=None,
                                               save_summaries_steps=None, save_summaries_secs=None) as sess:
            keras.backend.set_session(sess)

            num_states  = self.input_buffer.shape[1]
            state_shape = self.input_buffer.shape[2:]

            input_queue = self.input_queue
            ready_event = self.ready_event
            output_queue = self.output_queue

            input_buffer = self.input_buffer
            index_buffer = self.index_buffer
            policy_buffer = self.policy_buffer
            value_buffer = self.value_buffer

            policy = self.policy
            value = self.value
            x = self.x
            training_phase = self.training_phase
            ready_event.set()

            while True:
                size = input_queue.get()
                idx = index_buffer[:size]

                batch = input_buffer[idx]
                batch = batch.reshape(size * num_states, *state_shape)

                policy_batch, value_batch = sess.run([policy, value], {x: batch, training_phase: 0})
                ready_event.set()

                policy_batch = policy_batch.reshape(size, num_states, 12)
                value_batch = value_batch.reshape(size, num_states, 1)

                policy_buffer[idx, :, :] = policy_batch[:, :, :]
                value_buffer[idx, :, :] = value_batch[:, :, :]

                for worker in idx:
                    output_queue[worker].set()
