import numpy as np

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

# sigmoid = np.vectorize(sigmoid)
print(sigmoid([3, 4, 5]))               # [0.95257413 0.98201379 0.99330715]

