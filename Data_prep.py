import numpy as np
import h5py

#File to store functions for reading and formatting the data from the hdf5 files.

def read_data(filename):
    with h5py.File(filename, 'r') as d:
        data = d["Data"]
        response = d["response"]
        size = d["size"]
        save = np.array(data)
        Y = np.array(response)
        sizes = np.array(size)
    return save, Y, sizes

def partial_format(data, sizes, response, max_size): # Function to create 3D lattice.
    max_len = data.shape[1]
    data = data.astype(int)
    sizes = sizes.astype(int)
    dataa = np.repeat(data[:,:,0], sizes, axis=0)
    final = np.repeat(response, sizes)
    length = dataa.shape[0]
    lattice_map = np.zeros((length, 2*max_size-1, 2*max_size-1, 2*max_size-1, 1))
    location = np.zeros((length, max_len))
    count = 0
    for num, x in enumerate(sizes):
        for y in range(int(x)):
            location[count, y] = 1
            for i in range(y+1):
                lattice_map[count,int(max_len-1+data[num,i,1]),int(max_len-1+data[num,i,2]),int(max_len-1+data[num,i,3]),0] = dataa[count, i]
            count += 1
    return dataa, lattice_map, location, final

# location indicates the current place where the protein has been folded up to.
# dataa is the chain of amino acids
# lattice_map is the 3D array to hold the spatial locations of the folded protein
# final is the response or the evaluation energies for all of the partially folded proteins.