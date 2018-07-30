import numpy as np
import ctypes
from typing import Optional, Tuple

from multiprocessing import Process, Queue, Array, Event
from threading import Thread


class DataProcessCommand:
    """ A small class to hold constants for the possible data process commands. """
    Add = 0
    Get = 1
    Backup = 2
    Visit = 3
    Clear = 4

    TreeAdd = 5
    TreeGet = 6
    TreeClear = 7

    BothGet = 8
    BothAdd = 9


class BaseDatabase:
    ####################################################################################################
    # Private Commands
    ####################################################################################################
    def __add(self, idx, key):
        if key not in self.data:
            data = np.zeros((5, self.num_moves), dtype=np.float32)
            data[4, :] = self.policy_buffer[idx, :]
            self.data[key] = data

        self.add_ready_event[idx].set()

    def __get(self, key, idx):
        try:
            self.output_buffer[idx, :, :] = self.data[key][:]
        except KeyError:
            self.index_error_queue[idx].set()
        self.output_queue[idx].set()

    def __backup(self, key, action, last_value):
        try:
            store = self.data[key]
        except KeyError:
            return
        store[0, action] += 1
        store[1, action] += last_value
        store[2, action] = store[1, action] / store[0, action]
        # store[2, action] = max(store[2, action], last_value)
        store[3, action] -= 1

    def __visit(self, key, action):
        self.data[key][3, action] += 1

    def __clear(self):
        self.data.clear()
        for tree in self.trees:
            tree.clear()

    def __clear_tree(self, idx):
        self.trees[idx].clear()

    def __add_to_tree(self, idx, key):
        if self.single_tree:
            idx = 0
        self.trees[idx].add(key)

    def __is_in_tree(self, idx, key):
        if self.single_tree:
            tree_idx = 0
        else:
            tree_idx = idx

        self.tree_buffer[idx] = (key in self.trees[tree_idx])
        self.output_queue[idx].set()

    def __get_data_and_tree(self, idx, key):
        try:
            self.output_buffer[idx, :, :] = self.data[key][:]
        except KeyError:
            self.index_error_queue[idx].set()
        self.__is_in_tree(idx, key)

    def __add_data_and_tree(self, idx, key):
        self.__add_to_tree(idx, key)
        self.__add(idx, key)

    def run_command(self, idx, command, key, action, last_value):
        if command == 0:
            self.__add(idx, key)

        elif command == 1:
            self.__get(key, idx)

        elif command == 2:
            self.__backup(key, action, last_value)

        elif command == 3:
            self.__visit(key, action)

        elif command == 4:
            self.__clear()

        elif command == 5:
            self.__add_to_tree(idx, key)

        elif command == 6:
            self.__is_in_tree(idx, key)

        elif command == 7:
            self.__clear_tree(idx)

        elif command == 8:
            self.__get_data_and_tree(idx, key)

        elif command == 9:
            self.__add_data_and_tree(idx, key)


class DataProcess(Process, BaseDatabase):
    def __init__(self, num_moves: int, num_workers: int = 1, single_tree: bool = False,
                 synchronous: bool = True, num_action_threads: int = 16):
        """ This process manages the MCTS databases and trees used for asynchronous MCTS.

        Parameters
        ----------
        num_moves : int
            Number of moves your environment supports.
        num_workers : int
            Number of attached asyncronous workers.
        single_tree : bool
            Whether or not all of the workers share a single tree.
        synchronous : bool
            Whether or not to synchronize all database calls. Leaving this on will ensure correctness while
            False will allow operations to be performed faster.
        num_action_threads : int
            Number of asynchronous threads to use when synchronous=False. This is ignored otherwise.
        """
        super(DataProcess, self).__init__()

        self.num_moves = num_moves
        self.num_workers = num_workers
        self.single_tree = single_tree

        self.synchronous = synchronous
        self.num_action_threads = num_action_threads

        self.input_queue = Queue()
        self.kill_event = Event()
        self.output_queue = [Event() for _ in range(num_workers)]
        self.index_error_queue = [Event() for _ in range(num_workers)]

        # Output Buffer numpy array 0: N, 1: W, 2: Q, 3: V, 4: P
        self.__output_buffer_base = Array(ctypes.c_float, int(num_workers * num_moves * 5), lock=False)
        self.output_buffer = np.ctypeslib.as_array(self.__output_buffer_base)
        self.output_buffer = self.output_buffer.reshape(num_workers, 5, num_moves)

        self.__tree_buffer_base = Array(ctypes.c_bool, int(num_workers), lock=False)
        self.tree_buffer = np.frombuffer(self.__tree_buffer_base, np.bool, count=num_workers)

        self.__policy_buffer_base = Array(ctypes.c_float, int(num_workers * num_moves), lock=False)
        self.policy_buffer = np.ctypeslib.as_array(self.__policy_buffer_base)
        self.policy_buffer = self.policy_buffer.reshape(num_workers, num_moves)

        self.add_ready_event = [Event() for _ in range(num_workers)]
        for event in self.add_ready_event:
            event.set()

    def shutdown(self) -> None:
        """ Shutdown server. """
        if self.synchronous:
            self.input_queue.put((-1, -1, -1, -1, -1))
        else:
            for _ in range(self.num_action_threads):
                self.input_queue.put((-1, -1, -1, -1, -1))
            self.kill_event.set()

    def __str__(self):
        out = [super(DataProcess, self).__str__()]
        out.append("="*60)
        out.append("Synchronous: {}".format(self.synchronous))
        if not self.synchronous:
            out.append("Num Threads: {}".format(self.num_action_threads))
        out.append("-"*60)
        return "\n".join(out)

    ####################################################################################################
    # User Facing Commands
    ####################################################################################################
    def done_clear(self, idx: int) -> None:
        """ Clear this workers done event. This must be done before any of the GET commands are sent.

        Parameters
        ----------
        idx : int
            Worker Index.
        """
        self.output_queue[idx].clear()
        self.index_error_queue[idx].clear()

    def wait_for_done(self, idx: int) -> None:
        """ Block until this workers GET request has been complete.

        Parameters
        ----------
        idx : int
            Worker Index.
        """
        self.output_queue[idx].wait()

    def add(self, idx: int, key, policy) -> None:
        """ Add a state to the database. This function is non-blocking.

        Parameters
        ----------
        idx : int
            Worker Index.
        key : Hashable Object
            Hashable version of state to add.
        policy : np.ndarray
            Cache the policies for that state.
        """
        assert policy.shape[0] == self.num_moves and policy.ndim == 1, \
            "Policy must be 1-d vector with correct number of moves."

        self.add_ready_event[idx].wait()

        self.add_ready_event[idx].clear()
        self.policy_buffer[idx, :] = policy

        command = DataProcessCommand.Add
        self.input_queue.put((0, command, key, 0, 0))

    def get(self, idx: int, key) -> Optional[np.ndarray]:
        """ Get data associated with a given state. This command Blocks until finished.

        Parameters
        ----------
        idx : int
            Worker Index.
        key : Hashable Object
            Hashable version of state to add.

        Returns
        -------
        numpy.ndarray[float32]. The rows are as follows:
            N: Number of times the state has been visited.
            W: Total value of the state.
            Q: Normalized value of the state.
            V: Current Virtual loss of the state.
            P: Policy Cache
        """
        command = DataProcessCommand.Get
        self.done_clear(idx)
        self.input_queue.put((idx, command, key, 0, 0))

        self.wait_for_done(idx)

        if self.index_error_queue[idx].is_set():
            return None
        return self.output_buffer[idx]

    def backup(self, key, action: int, last_value: float) -> None:
        """ Perform a MCTS backup operations for a given state and action. This function is non-blocking.

        Parameters
        ----------
        key : Hashable Object
            Hashable version of state to add.
        action : int
            Action performed at that state.
        last_value : float
            Value of leaf node during simulation.
        """
        command = DataProcessCommand.Backup
        self.input_queue.put((0, command, key, action, last_value))

    def visit(self, key, action: int) -> None:
        """ Increment virtual loss value of a state, action pair. This function is non-blocking.

        Parameters
        ----------
        key : Hashable Object
            Hashable version of state to add.
        action : int
            Action performed at that state.
        """
        command = DataProcessCommand.Visit
        self.input_queue.put((0, command, key, action, 0))

    def clear(self) -> None:
        """ Clear the database. This function is non-blocking."""
        command = DataProcessCommand.Clear
        self.input_queue.put((0, command, 0, 0, 0))

    def tree_add(self, idx: int, key) -> None:
        """ Add a new state to this workers tree. If single_tree is on, then it will add the state to the main tree.

        This function is non-blocking.

        Parameters
        ----------
        idx : int
            Worker Index. Ignored if single_tree=True.
        key : Hashable Object
            Hashable version of state to add.
        """
        command = DataProcessCommand.TreeAdd
        self.input_queue.put((idx, command, key, 0, 0))

    def tree_get(self, idx: int, key) -> bool:
        """ Get wether or not a given state is in this worker's tree. This function blocks until the result is done.

        Parameters
       ----------
        idx : int
            Worker Index. Ignored if single_tree=True.
        key : Hashable Object
            Hashable version of state to add.

        Returns
        -------
        bool
            True is the state is present in the tree.
        """
        command = DataProcessCommand.TreeGet

        self.done_clear(idx)
        self.input_queue.put((idx, command, key, 0, 0))

        self.wait_for_done(idx)
        return self.tree_buffer[idx]

    def tree_clear(self, idx: int) -> None:
        """ Clear this worker's tree. This function is non-blocking.

        Parameters
        ----------
        idx : int
            Worker Index. Ignored if single_tree=True.
        """
        command = DataProcessCommand.TreeClear
        self.input_queue.put((idx, command, 0, 0, 0))

    def both_get(self, idx: int, key) -> Tuple[bool, Optional[np.ndarray]]:
        """ Get whether or not the state is in the tree and the data associated with the state.

        If no data is available for this state, it will return a None object.
        This function block until a result is ready.

        Parameters
        ----------
        idx : int
            Worker Index.
        key : Hashable Object
            Hashable version of state to add.

        Returns
        -------
        bool
            True is the state is present in the tree.

        numpy.ndarray[float32] Or NONE if state is not in database. The rows are as follows:
            N: Number of times the state has been visited.
            W: Total value of the state.
            Q: Normalized value of the state.
            V: Current Virtual loss of the state.
            P: Policy cache
        """
        command = DataProcessCommand.BothGet

        self.done_clear(idx)
        self.input_queue.put((idx, command, key, 0, 0))

        self.wait_for_done(idx)

        if self.index_error_queue[idx].is_set():
            output = None
        else:
            output = self.output_buffer[idx]

        return self.tree_buffer[idx], output

    def both_add(self, idx: int, key, policy) -> None:
        """ Add a state to both this workers tree and the database. This function is non-blocking.

        Parameters
        ----------
        idx : int
            Worker Index.
        key : Hashable Object
            Hashable version of state to add.
        policy : np.ndarray
            Cache the policies for that state.
        """
        assert policy.shape[0] == self.num_moves and policy.ndim == 1, \
            "Policy must be 1-d vector with correct number of moves."

        self.add_ready_event[idx].wait()

        self.add_ready_event[idx].clear()
        self.policy_buffer[idx, :] = policy

        command = DataProcessCommand.BothAdd
        self.input_queue.put((idx, command, key, 0, 0))

    def __initialize_data(self):
        self.data = {}
        self.trees = [set() for _ in range(self.num_workers)]

    def run(self):
        self.__initialize_data()

        if not self.synchronous:
            threads = []
            for i in range(self.num_action_threads):
                thread = DatabaseWorker(self.single_tree, self.num_workers, self.num_moves,
                                        self.input_queue, self.output_queue, self.index_error_queue,
                                        self.output_buffer, self.tree_buffer, self.policy_buffer,
                                        self.add_ready_event, self.data, self.trees, i)
                thread.start()
                threads.append(thread)

            self.kill_event.wait()

        else:
            while True:
                idx, command, key, action, last_value = self.input_queue.get()
                if idx < 0:
                    break
                self.run_command(idx, command, key, action, last_value)


class DatabaseWorker(Thread, BaseDatabase):
    def __init__(self, single_tree, num_workers, num_moves, input_queue, output_queue, index_error_queue,
                 output_buffer, tree_buffer, policy_buffer, add_ready_event, data, trees, thread_number):
        super(DatabaseWorker, self).__init__()

        self.single_tree = single_tree
        self.num_workers = num_workers
        self.num_moves = num_moves
        self.thread_number = thread_number

        self.input_queue = input_queue
        self.output_queue = output_queue
        self.index_error_queue = index_error_queue
        self.output_buffer = output_buffer
        self.tree_buffer = tree_buffer
        self.policy_buffer = policy_buffer
        self.add_ready_event = add_ready_event

        self.data = data
        self.trees = trees

    def run(self):
        while True:
            idx, command, key, action, last_value = self.input_queue.get()
            if idx < 0:
                break
            self.run_command(idx, command, key, action, last_value)
