from multiprocessing import Process, Queue, Event
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Callable

class DistributedNetworkConfig:
    def __init__(self, learning_rate=0.01,
                 policy_weight=1.0,
                 training_batch_size=64,
                 tensorboard_log=False, **kwargs):
        self.learning_rate = learning_rate
        self.policy_weight = policy_weight
        self.training_batch_size = training_batch_size
        self.tensorboard_log = tensorboard_log

class DistributedNetworkProcess(Process):
    def __init__(self, make_network: Callable[[], keras.Model],
                 session_config: tf.ConfigProto,
                 task_index: int,
                 parameter_server: bool,
                 cluster_spec: tf.train.ClusterSpec,
                 input_queue: Queue,
                 ready_event: Event,
                 output_ready: [Event],
                 input_buffer: np.ndarray,
                 index_buffer: np.ndarray,
                 policy_buffer: np.ndarray,
                 value_buffer: np.ndarray,
                 **kwargs):
        """ Class for managing a distributed tensorflow model.

        This class creates the TF graphs and the distributed server.

        Parameters
        ----------
        make_network : () -> keras.Model
            Function defining how to construct your network.
        session_config : tf.ConfigProto
            Tensorflow session config object.
        task_index : int
            Index of this worker on the cluster spec.
        parameter_server : bool
            Whether or not this instance is a parameter server.
        cluster_spec : tf.train.ClusterSpec
            Cluster Spec containing the paths for all workers in the cluster.
        input_queue : Queue
            This networks input queue.
        ready_event : Event
            This networks ready event.
        output_ready : [Event]
            Output events for all connected workers.
        input_buffer : np.ndarray
            Input buffer for storing worker inputs
        index_buffer : np.ndarray
            This workers index buffer, this is where it gets prediction requests from.
        policy_buffer : np.ndarray
            Output buffer.
        value_buffer : np.ndarray
            Output buffer.
        kwargs
            Optional arguments that are passed to DistributedNetworkConfig.
        """
        super(DistributedNetworkProcess, self).__init__()

        self.make_network = make_network
        self.session_config = session_config
        self.network_config = DistributedNetworkConfig(**kwargs)

        self.task_index = task_index
        self.cluster_spec = cluster_spec
        self.parameter_server = parameter_server

        self.input_queue = input_queue
        self.ready_event = ready_event
        self.output_ready = output_ready

        self.input_buffer = input_buffer
        self.index_buffer = index_buffer
        self.policy_buffer = policy_buffer
        self.value_buffer = value_buffer

    def __initialize_network(self):
        """ Create Tensorflow graph. """
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

            self.summary_op = None
            if self.network_config.tensorboard_log:
                with tf.name_scope("Loss"):
                    tf.summary.scalar('policy_loss', self.policy_loss)
                    tf.summary.scalar('value_loss', self.value_loss)
                    tf.summary.scalar('total_loss', self.total_loss)

                for layer in self.model.layers:
                    with tf.name_scope(layer.name):
                        for weight in layer.weights:
                            with tf.name_scope(weight.name.split("/")[-1].split(":")[0]):
                                tf.summary.histogram('histogram', weight)

                self.summary_op = tf.summary.merge_all()

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

        # Create a monitored session for communication between network workers.
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=self.task_index == 0,
                                               hooks=hooks, chief_only_hooks=chief_only_hooks,
                                               config=self.session_config,
                                               save_checkpoint_secs=None, save_checkpoint_steps=None,
                                               save_summaries_steps=None, save_summaries_secs=None) as sess:
            keras.backend.set_session(sess)

            # Store Variables locally for faster access
            num_states = self.input_buffer.shape[1]
            state_shape = self.input_buffer.shape[2:]

            input_queue = self.input_queue
            ready_event = self.ready_event
            output_ready = self.output_ready

            input_buffer = self.input_buffer
            index_buffer = self.index_buffer
            policy_buffer = self.policy_buffer
            value_buffer = self.value_buffer

            policy = self.policy
            value = self.value
            x = self.x
            training_phase = self.training_phase

            # Ready to predict
            ready_event.set()

            while True:
                # Wait for a new request from manager.
                size = input_queue.get()
                idx = index_buffer[:size].copy()

                # Create the appropriate input batch
                batch = input_buffer[idx]
                batch = batch.reshape(size * num_states, *state_shape)

                # Predict from the network.
                policy_batch, value_batch = sess.run([policy, value], {x: batch, training_phase: 0})

                # At this stage, we're done with the input and index buffer. So the manager can place more inputs.
                ready_event.set()

                # Reshape and output results
                policy_batch = policy_batch.reshape(size, num_states, 12)
                value_batch = value_batch.reshape(size, num_states, 1)

                policy_buffer[idx, :, :] = policy_batch[:, :, :]
                value_buffer[idx, :, :] = value_batch[:, :, :]

                # Signal to workers that their results are ready.
                for worker in idx:
                    output_ready[worker].set()

class DistributedTrainingProcess(DistributedNetworkProcess):
    def __init__(self, make_network: Callable[[], keras.Model],
                 session_config: tf.ConfigProto,
                 task_index: int,
                 cluster_spec: tf.train.ClusterSpec,
                 input_queue: Queue,
                 ready_event: Event,
                 training_buffer: np.ndarray,
                 policy_target_buffer: np.ndarray,
                 value_target_buffer: np.ndarray,
                 **kwargs):
        super(DistributedTrainingProcess, self).__init__(make_network=make_network,
                                                         session_config=session_config,
                                                         task_index=task_index,
                                                         parameter_server=False,
                                                         cluster_spec=cluster_spec,
                                                         input_queue=input_queue,
                                                         ready_event=ready_event,
                                                         output_ready=None,
                                                         input_buffer=None,
                                                         index_buffer=None,
                                                         policy_buffer=None,
                                                         value_buffer=None,
                                                         **kwargs)

        self.training_buffer = training_buffer
        self.policy_target_buffer = policy_target_buffer
        self.value_target_buffer = value_target_buffer

    def run(self):
        server = tf.train.Server(self.cluster_spec, job_name="worker", task_index=self.task_index,
                                 config=self.session_config)

        self.__initialize_network()

        # Add hooks if necessary
        hooks = None
        chief_only_hooks = None

        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=False,
                                               hooks=hooks, chief_only_hooks=chief_only_hooks,
                                               config=self.session_config,
                                               save_checkpoint_secs=None, save_checkpoint_steps=None,
                                               save_summaries_steps=None, save_summaries_secs=None) as sess:
            keras.backend.set_session(sess)

            input_queue = self.input_queue
            ready_event = self.ready_event

            training_buffer = self.training_buffer
            policy_target_buffer = self.policy_target_buffer
            value_target_buffer = self.value_target_buffer

            batch_size = self.network_config.training_batch_size

            ready_event.set()

            while True:
                size = input_queue.get()

                train_data = training_buffer[:size]
                policy_targets = policy_target_buffer[:size]
                value_targets = value_target_buffer[:size]

                num_batches = (size // batch_size) + 1

                for batch in range(num_batches):
                    low_idx = batch * batch_size
                    high_idx = (batch + 1) * batch_size

                    run_list = [self.train_op, self.policy_loss, self.value_loss, self.global_step]
                    feed_dict = {self.x: train_data[low_idx:high_idx],
                                 self.policy_target: policy_targets[low_idx:high_idx],
                                 self.value_target: value_targets[low_idx:high_idx]}

                    _, ploss, vloss, step = sess.run(run_list, feed_dict)

                    print("Step {}: Value Loss = {} --- Policy Loss = {}".format(step, vloss, ploss))

                ready_event.set()

