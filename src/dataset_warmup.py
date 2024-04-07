# Process A: DataStreamer.py

from multiprocessing import shared_memory
from scipy.io import arff
from time import sleep
import numpy as np
import time


def create_shared(data, name):
    print(data.shape)
    d_size = np.dtype(np.float64).itemsize * np.prod(data.shape)

    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    dst = np.ndarray(shape=data.shape, dtype=np.float64, buffer=shm.buf)
    dst[:] = data[:]

    return shm

# def get_shared():
#     print("hi")
#     shm = shared_memory.SharedMemory(name='npshared')
#     print("hi")
#     np_array = np.ndarray(shape=(1000,34), dtype=np.float64, buffer=shm.buf)
#     print("hi")
#     ret = np.ndarray(shape=(1000,34), dtype=np.float64)
#     ret[:] = np_array[:]
#     return ret

def main():
    print("Begin Loading Dataset Data")
    start = time.time()
    dataset, meta = arff.loadarff("../dataset/dataset.arff")
    print("End Loading Data")
    data = np.array(dataset.tolist(), dtype=np.float64)
    shm = create_shared(data, 'npfull')
    end = time.time()
    print("Warmup time:", end - start)


    print("Begin Loading pruned Data")
    start = time.time()
    dataset, meta = arff.loadarff("../dataset/pruned.arff")
    print("End Loading Data")
    data = np.array(dataset.tolist(), dtype=np.float64)
    shm = create_shared(data, 'nppruned')
    end = time.time()
    print("Warmup time:", end-start)

    #ensure data are persistent
    while True:
        if 'q' in input("Enter q to quit"):
            break

if __name__ == "__main__":
    main()


