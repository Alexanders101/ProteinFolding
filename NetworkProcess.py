from multiprocessing import Process, Queue, Array, Event
import keras
import tensorflow as tf
import ctypes
import numpy as np

class NetworkProcess(Process):
    def __init__(self, make_model, state_shape, num_moves, num_states=1, num_workers=1, batch_size=None,
                 session_config=None):
        super(NetworkProcess, self).__init__()

        self.make_model = make_model
        self.session_config = session_config

        self.state_shape = state_shape
        self.num_moves = num_moves

        self.num_states = num_states
        self.num_workers = num_workers

        if batch_size is None:
            batch_size = num_states * num_workers
        self.batch_size = batch_size

        self.__ready_queue = Queue(maxsize=1)

        self.input_queue = Queue(maxsize=num_workers)
        self.output_queue = [Event() for _ in range(num_workers)]
        # self.output_queue = [Queue(1) for _ in range(num_workers)]

        self.__input_buffer_base = Array(ctypes.c_int64, int(num_workers * num_states * np.prod(state_shape)), lock=False)
        self.input_buffer = np.ctypeslib.as_array(self.__input_buffer_base)
        self.input_buffer = self.input_buffer.reshape(num_workers, num_states, *state_shape)

        self.__policy_buffer_base = Array(ctypes.c_float, int(num_workers * num_states * num_moves), lock=False)
        self.policy_buffer = np.ctypeslib.as_array(self.__policy_buffer_base)
        self.policy_buffer = self.policy_buffer.reshape(num_workers, num_states, num_moves)

        self.__value_buffer_base = Array(ctypes.c_float, int(num_workers * num_states), lock=False)
        self.value_buffer = np.ctypeslib.as_array(self.__value_buffer_base)
        self.value_buffer = self.value_buffer.reshape(num_workers, num_states, 1)


    def predict(self, idx, states):
        assert self.is_alive(), "Network has not been started or has already been shutdown."
        return self._predict_unsafe(idx, states)

    def _predict_unsafe(self, idx, states):
        self.input_buffer[idx, :] = states[:]
        self.output_queue[idx].clear()
        self.input_queue.put_nowait(idx)

        self.output_queue[idx].wait()
        return self.policy_buffer[idx], self.value_buffer[idx]

    def shutdown(self):
        self.input_queue.put(-1)

    def ready(self):
        return not self.__ready_queue.empty()

    def __initialize_network(self):
        self.session = tf.Session(config=self.session_config)
        keras.backend.set_session(self.session)

        self.model = self.make_model()
        self.model.compile('adam', loss=['categorical_crossentropy', 'MSE'])

    def run(self):
        self.__initialize_network()
        ids = np.zeros(self.num_workers, dtype=np.uint16)
        input_buffer = self.input_buffer.reshape(self.num_workers * self.num_states, *self.state_shape)

        self.__ready_queue.put(True)
        while True:
            ids[0] = self.input_queue.get()
            if ids[0] >= self.num_workers:
                break

            size = 1
            while size < self.num_workers and not self.input_queue.empty():
                ids[size] = self.input_queue.get()
                size += 1

            sorted_idx = np.where(np.bincount(ids[:size]))[0]
            batch = self.input_buffer[sorted_idx]
            batch = batch.reshape(size * self.num_states, *self.state_shape)

            policy, value = self.model.predict(batch, batch_size=self.batch_size)
            policy = policy.reshape(size, self.num_states, 12)
            value = value.reshape(size, self.num_states, 1)

            for i in range(size):
                idx = sorted_idx[i]
                self.policy_buffer[idx, :, :] = policy[i, :, :]
                self.value_buffer[idx, :, :] = value[i, :, :]

                self.output_queue[idx].set()

            # policy, value = self.model.predict(input_buffer, batch_size=self.batch_size)
            # policy = policy.reshape(self.num_workers, self.num_states, 12)
            # value = value.reshape(self.num_workers, self.num_states, 1)
            #
            # for i in range(size):
            #     idx = ids[i]
            #     self.policy_buffer[idx, :, :] = policy[idx, :, :]
            #     self.value_buffer[idx, :, :] = value[idx, :, :]
            #
            #     self.output_queue[idx].put(1)