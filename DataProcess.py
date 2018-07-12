import numpy as np
import ctypes

from multiprocessing import Process, Queue, Array, Event
from concurrent.futures import ThreadPoolExecutor

class DataProcess(Process):
    def __init__(self, num_moves, num_workers=1, synchronous=True, num_action_threads=16):
        super(DataProcess, self).__init__()

        self.num_moves = num_moves
        self.num_workers = num_workers

        self.synchronous = synchronous
        self.num_action_threads = num_action_threads

        self.input_queue = Queue()
        self.output_queue = [Event() for _ in range(num_workers)]

        # Output Buffer numpy array 0: N, 1: W, 2: Q, 3: V
        self.__output_buffer_base = Array(ctypes.c_float, int(num_workers * num_moves * 4), lock=False)
        self.output_buffer = np.ctypeslib.as_array(self.__output_buffer_base)
        self.output_buffer = self.output_buffer.reshape(num_workers, 4, num_moves)

    def shutdown(self):
        self.input_queue.put(-1)

    def add(self, key):
        self.input_queue.put((0, 0, key, 0, 0))

    def get(self, idx, key):
        self.output_queue[idx].clear()
        self.input_queue.put((idx, 1, key, 0, 0))

        self.output_queue[idx].wait()
        return self.output_buffer[idx]

    def backup(self, key, action, last_value):
        self.input_queue.put((0, 2, key, action, last_value))

    def visit(self, key, action):
        self.input_queue.put((0, 3, key, action, 0))

    def clear(self):
        self.input_queue.put((0, 4, 0, 0, 0))

    def __initialize_data(self):
        self.data = {}
        self.thread_pool = ThreadPoolExecutor(self.num_action_threads)

    def __add(self, key):
        if key not in self.data:
            self.data[key] = np.zeros((4, self.num_moves), dtype=np.float32)

    def __get(self, key, idx):
        self.output_buffer[idx, :, :] = self.data[key][:]
        self.output_queue[idx].set()

    def __backup(self, key, action, last_value):
        store = self.data[key]
        store[0, action] += 1
        store[1, action] += last_value
        store[2, action] = max(store[2, action], last_value)
        store[3, action] -= 1

    def __visit(self, key, action):
        self.data[key][3, action] += 1

    def __clear(self):
        self.data.clear()

    def __run_command(self, idx, command, key, action, last_value):
        if command == 0:
            self.__add(key)

        elif command == 1:
            self.__get(key, idx)

        elif command == 2:
            self.__backup(key, action, last_value)

        elif command == 3:
            self.__visit(key, action)

        elif command == 4:
            self.__clear()

    def run(self):
        self.__initialize_data()

        while True:
            idx, command, key, action, last_value = self.input_queue.get()
            if idx >= self.num_workers:
                break

            if self.synchronous:
                self.__run_command(idx, command, key, action, last_value)
            else:
                self.thread_pool.submit(self.__run_command, idx, command, key, action, last_value)
