# Process A: DataStreamer.py

import ray
from scipy.io import arff
from time import sleep

@ray.remote
class DataStore:
    def __init__(self):
        self.data = None

    def set_data(self, ndarray):
        self.data = ndarray

    def get_data(self):
        return self.data

if __name__ == "__main__":
    ray.init()  # Initialize Ray

    dataset, meta = arff.loadarff("../dataset/pruned.arff")

    # Create an instance of the DataStore actor
    data_store = DataStore.remote()

    # Generate some random data (replace this with your own ndarray creation)
    data_array = dataset

    # Set the data in the DataStore actor
    ray.get(data_store.set_data.options(name="some_name").remote(data_array))

    # Define a function to retrieve the data from the DataStore actor
    @ray.remote
    def retrieve_data(data_store_actor):
        return ray.get(data_store_actor.get_data.remote())

    # Spawn a process to retrieve the data
    result = ray.get(retrieve_data.remote(data_store))

    print(result, data_store)

    while True:
        sleep(10)
        pass


