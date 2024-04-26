from util import *
import torch

# Apple is 0
# Banana is 0.1
# Cherry is 0.2
actual = [0, 0, 0.1, 0.1, 0.2, 0.2, 0, 0.1, 0.2, 0, 0.1, 0.2, 0.1, 0.2,0]
predicted = [0, 0.2, 0.1, 0.1, 0.2, 0, 0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, 0.2, 0]

print (len(actual))
print(len(predicted))

actual = onehot_encoding(torch.tensor(actual))
predicted = onehot_encoding(torch.tensor(predicted))

matrix = confusion(predicted, actual)

print(matrix[:3, :3])

rec = recall(matrix)
print(rec)

prec = precision(matrix)
print(prec)

acc = accuracy(predicted, actual)
print(acc)

