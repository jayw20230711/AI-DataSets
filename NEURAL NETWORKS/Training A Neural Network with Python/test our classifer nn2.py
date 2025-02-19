"""
4. Prg
"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from neural_networks2 import NeuralNetwork

n_samples = 500
blob_centers = ([2, 6], [6, 2], [7, 7])
n_classes = len(blob_centers)
data, labels = make_blobs(n_samples=n_samples,
                          centers=blob_centers,
                          random_state=7)

"""
We need a one-hot representation for each label. So the labels are represented as
Label   One-Hot Representation
0           (1, 0, 0)
1           (0, 1, 0)
2           (0, 0, 1)
We can easily change the labels with the following commands:
"""
labels = np.arange(n_classes) == labels.reshape(labels.size, 1)
labels = labels.astype(np.float64)

res = train_test_split(data, labels,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)
train_data, test_data, train_labels, test_labels = res

simple_network = NeuralNetwork(no_of_in_nodes=2,
                               no_of_out_nodes=3,
                               no_of_hidden_nodes=5,
                               learning_rate=0.1,
                               bias=1)

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

print(simple_network.evaluate(train_data, train_labels))
