from Lattice2 import Lattice2

import numpy as np

import csv

import h5py

max_size = 25
sample_size = 50000
max_aa = 40
directions = False

file_store = 'small_data_res.h5'

with h5py.File(file_store, 'w') as h5file:

    h5file.create_dataset("Data", shape=(sample_size, max_aa, 4))
    h5file.create_dataset("response", shape=(sample_size,))
    h5file.create_dataset("size", shape=(sample_size,))
    h5file.create_dataset("max_size", shape=(1,))
    if directions:
        h5file.create_dataset("direction", shape=(sample_size, max_aa))
    h5file["max_size"][0] = max_size
    for x in np.arange(sample_size):
        temp = Lattice2(int(np.random.choice(np.arange(5, max_aa), size=1)), 3, max_size=max_size)
        results = temp.make_rand()
        together = np.zeros((max_aa, 4))
        together[:temp.size,:] = np.reshape(np.concatenate((np.reshape(temp.aa_string, (temp.size, 1)), temp.lattice), axis=1), (temp.size, 4))
        h5file["Data"][x,:,:] = together
        h5file["response"][x] = results
        h5file["size"][x] = temp.size
        if directions:
            use = mp.zeros(max_aa)
            use[:temp.size-1] = temp.directions
            h5file["direction"][x, :] = use
        if x%1000==0:
            print(x)
