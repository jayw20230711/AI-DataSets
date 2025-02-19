"""
2. Prg
We assume that you save the previous code in a file called neural_networks1.py . We will use it under
this name in the coming examples.
To test this neural network class we need train and test data. We create the data with make_blobs from
sklearn.datasets
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from neural_networks1 import NeuralNetwork

n_samples = 500
blob_centers = ([2, 6], [6, 2], [7, 7])   # 3
n_classes = len(blob_centers)
# print("n_classes : ", n_classes)      # 3
data, labels = make_blobs(n_samples=n_samples,
                          centers=blob_centers,
                          random_state=7)

# print("\ndata : ", data)          # [[ 7.59267971  9.3518673 ] ...
# print("\nlabels : ", labels)      # [2 2 1 0 2 0 1 2
"""
Let us visualize the previously created data:
"""
colours = ('green', 'red', "yellow")
fig, ax = plt.subplots()

for n_class in range(n_classes):
    ax.scatter(data[labels == n_class][:, 0],
               data[labels == n_class][:, 1],
               c=colours[n_class],
               s=40,
               label=str(n_class))

plt.show()

# The labels are wrongly represented. They are in a one-dimensional vector:
print("\nlabels[:7] : ", labels[:7])       # [2 2 1 0 2 0 1]

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

# print("\nOne-hot arrangement: ")
# print(labels[:7])

"""
We are ready now to create a train and a test data set: 
"""
res = train_test_split(data, labels,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)

train_data, test_data, train_labels, test_labels = res

print("\nTrain_labels : ", train_labels[:10])
print("\ntrain_data : ", train_data[:10])
# print("len of train_data", len(train_data))     # 400
"""
We create a neural network with two input nodes, and three output nodes. One output node for each class:
from neural_networks1 import NeuralNetwork
"""
simple_network = NeuralNetwork(no_of_in_nodes=2,
                               no_of_out_nodes=3,
                               no_of_hidden_nodes=5,
                               learning_rate=0.3)

"""
The next step consists in training our network with the data and labels from our training samples:
"""
for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

out = simple_network.evaluate(train_data, train_labels)
print("\nOut : ", out)          # (390, 10)
