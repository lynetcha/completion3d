'''
Parallel data loading classes
'''

import sys
import time
import gc  # garbage collector
import numpy as np
import traceback
import six.moves.queue as queue
from multiprocessing import Process, Event


def print_error(func):
    '''Flush out error messages. Mainly used for debugging separate processes'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            traceback.print_exception(*sys.exc_info())
            sys.stdout.flush()

    return func_wrapper


class DataProcess(Process):

    def __init__(self, data_queue, data_paths, data_shape, batch_size, repeat=True):
        '''
        data_queue : Multiprocessing queue
        data_paths : list of data and label pair used to load data
        data_shape : shape of data
        repeat : if set True, return data until exit is set
        '''
        Process.__init__(self)
        # Queue to transfer the loaded mini batches
        self.data_queue = data_queue
        self.data_paths = data_paths
        self.num_data = len(data_paths)
        self.repeat = repeat

        # Tuple of data shape
        self.data_shape = data_shape
        self.batch_size = batch_size
        self.gc_freq = 1000
        self.exit = Event()
        self.shuffle_db_inds()


    def shuffle_db_inds(self):
        # Randomly permute the training roidb
        if self.repeat:
            self.perm = np.random.permutation(np.arange(self.num_data))
        else:
            self.perm = np.arange(self.num_data)
        self.cur = 0

    def get_next_minibatch(self, batch_size=0):
        batch_size = batch_size or self.batch_size
        if (self.cur + batch_size) >= self.num_data and self.repeat:
            self.shuffle_db_inds()

        db_inds = self.perm[self.cur:min(self.cur + batch_size, self.num_data)]
        self.cur += batch_size
        return db_inds

    def shutdown(self):
        self.exit.set()

    @print_error
    def run(self):
        iteration = 0
        # Run the loop until exit flag is set
        while not self.exit.is_set() and self.cur <= self.num_data:
            # Ensure that the network sees (almost) all data per epoch
            db_inds = self.get_next_minibatch()
            if len(db_inds) == 0:
                break

            batch = []
            # Load data
            for batch_id, db_ind in enumerate(db_inds):
                # Call loading functions
                data = self.load_data(self.data_paths[db_ind])
                batch.append(data)

            self.data_queue.put(self.collate(batch), block=True)
            iteration += 1

            # For SLURM failure
            if iteration % self.gc_freq == 0:
                gc.collect()

    def load_datum(self, path):
        pass

    def load_label(self, path):
        pass


def get_while_running(data_processes, data_queue, sleep_time=0):
    while True:
        time.sleep(sleep_time)
        try:
            input_dict = data_queue.get_nowait()
        except queue.Empty:
            alive = True
            if type(data_processes) is list:
                for data_process in data_processes:
                    alive = alive and data_process.is_alive()
            else:
                alive = data_processes.is_alive()
            # Break if data processes are all dead
            if not alive:
                break
            else:
                continue
        yield input_dict


def kill_data_processes(queue, processes):
    print('Signal processes')
    for p in processes:
        p.shutdown()

    time.sleep(0.5)
    while not queue.empty():
        queue.get(False)
        time.sleep(0.5)
    # Close the queue
    queue.close()

    for p in processes:
        p.terminate()

def test_process():
    from multiprocessing import Queue

    data_queue = Queue(5)
    data_processes = []
    for i in range(5):
        data_processes.append(DataProcess(data_queue, train=True))
        data_processes[-1].start()

    count = 0
    data_paths = data_processes[-1].data_paths
    for data_dict in get_while_running(data_processes, data_queue, 0.5):
        for data_ind in range(len(data_dict[list(data_dict.keys())[0]])):
            print(data_paths[count], count)
            for k, v in data_dict.items():
                print(k, v.shape, v.dtype)
            count += 1

    print(count, len(data_paths))
    kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    test_process()
