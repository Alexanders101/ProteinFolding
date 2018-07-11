"""
Author: Alexander Shmakov
Date: 2018-02-13
Version: 1.0

This module defines the PredictorThread class which is responsible for the
asynchronous predictions that are required for the AsyncMCTS.

"""
from threading import Thread
import numpy as np


class PredictorThread(Thread):
    def __init__(self, model, prediction_queue, result_queues, state_shape, num_states, batch_size=8, min_percent=0.5):
        """
        This object is responsible for asynchronously managing predictions to a Keras model

        Parameters
        ----------
        model : keras.Model
            The model to predict from: State-> (Policy, Value)
        prediction_queue : Queue
            The queue from which inputs will be pulled from
        result_queues : [Queue]
            A list of queues to which results will be put into
        state_shape : tuple(int)
            State of a single state
        num_states : int
            Number of states that you will feed in at once
        batch_size : int
            Maximum number of requests to process at once

        Notes
        -----
        In order to use this class, another piece of code must add its desired inputs to the
        predication queue. When adding to the prediction queue, always add a tuple (index, input).
        The index specifies which result_queue the output will be placed in. The shape of the input must be
        (num_states, *state_shape).

        """
        super(PredictorThread, self).__init__()
        self.setDaemon(True)

        self.model = model
        self.prediction_queue = prediction_queue
        self.result_queues = result_queues

        self.state_shape = state_shape
        self.num_states = num_states
        self.batch_size = batch_size
        self.min_num = int(np.round(min_percent * batch_size))

        self.exit_flag = False

    def run(self):
        # Create permanent buffers
        ids = np.zeros(self.batch_size, dtype=np.uint16)
        states = np.zeros((self.batch_size, self.num_states) + self.state_shape, dtype=np.float32)

        while not self.exit_flag:
            ids[0], states[0] = self.prediction_queue.get()

            size = 1
            while (size < self.batch_size and not self.prediction_queue.empty()):
                ids[size], states[size] = self.prediction_queue.get()
                size += 1

            batch = np.reshape(states,
                               newshape=(self.batch_size * self.num_states,) + self.state_shape)

            policy, value = self.model.predict(batch[:size * self.num_states])

            for i in range(size):
                beg = i * self.num_states
                end = beg + self.num_states
                self.result_queues[ids[i]].put((policy[beg:end], value[beg:end]))
