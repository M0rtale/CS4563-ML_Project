from scipy.io import arff
import numpy as np


# dataset = arff.load(open('../dataset/dataset.arff', 'rb'))
dataset, meta = arff.loadarff('../dataset/dataset.arff')

data = np.array(dataset.tolist())

print(data.shape)
