# Process A: DataStreamer.py

import ray
from scipy.io import arff
from time import sleep

@ray.remote
class DataStreamer:
    def __init__(self, file_path):
        self.dataset, self.meta = arff.loadarff(file_path)

    def get_data(self):
        return self.dataset, self.meta

if __name__ == "__main__":
    ray.init()  # Initialize Ray

    # Assuming the dataset file is in the correct location
    file_path = '../dataset/pruned.arff'
    data_streamer = DataStreamer.remote(file_path)

    # Optionally, register the actor under a global name for easy access across processes
    #ray.register_actor("global_data_streamer", data_streamer)
    print("DataStreamer is ready and serving data.")

    while True:
        sleep(10)
