import ProteinEnv

import numpy as np

import csv

import h5py


sample_size = 10000 # Number of training examples to make
max_aa = 50 # Max length of amino acids
directions = False # Boolean to include a direction which indicates where the previous
                    # amino acid was located relative to the current one.

file_store = 'partial_fold_training.h5'


with h5py.File(file_store, 'w') as h5file:

    h5file.create_dataset("Data", shape=(sample_size*(max_aa-1), 5, max_aa))
    h5file.create_dataset("response", shape=(sample_size*(max_aa-1),))
    h5file.create_dataset("policy", shape=(sample_size*(max_aa-1), 12))
    start = ProteinEnv.NPProtein(max_aa)
    for x in np.arange(sample_size):
        state = start.random_state()
        big_state, policy = start.random_moves(state, policy=True)
        energy = start.reward(big_state[-1])
        response = np.repeat(energy, max_aa-1)
        h5file["Data"][x*(max_aa-1):(x+1)*(max_aa-1)] = big_state[:-1]
        h5file["response"][x*(max_aa-1):(x+1)*(max_aa-1)] = start.reward(state)
        h5file["policy"][x*(max_aa-1):(x+1)*(max_aa-1)] = policy
        if x%1000==0:
            print(x)
