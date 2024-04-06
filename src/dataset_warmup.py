# Process A: DataStreamer.py

import ray
from scipy.io import arff
from time import sleep

# @ray.remote
# class DataStreamer:
#     def __init__(self, file_path):
#         self.dataset, self.meta = arff.loadarff(file_path)

#     def get_data(self):
#         return self.dataset, self.meta

if __name__ == "__main__":
    ray.init()  # Initialize Ray

    dataset, meta = arff.loadarff("../dataset/pruned.arff")

    ray.put(dataset)

    ray.put(meta)

    print("DataStreamer is ready and serving data.")

    while True:
        sleep(10)
