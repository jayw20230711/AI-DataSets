""""
EXERCISE
We created in the chapter "Data Creation" a file strange_flowers.txt in the folder data . Create a
Neural Network to classify the 'flowers':
The data looks like this:
0.000,240.000,100.000,3.020
253.000,99.000,13.000,3.875
202.000,107.000,6.000,4.1
"""
import numpy as np
from sklearn import preprocessing        # for scale
from sklearn.model_selection import train_test_split
from neural_networks2 import NeuralNetwork

c = np.loadtxt("strange_flowers.txt", delimiter=" ")

data = c[:, :-1]
n_classes = data.shape[1]
# print(data.shape)           # (795, 4)
labels = c[:, -1]
# print(labels.shape)            # (795,)
# print(labels)             # ex:  1 , 2, 3,  4  values in the last column
# print(data[:5])

# mapping to a one hot scenario
labels = np.arange(n_classes) == labels.reshape(labels.size, 1)   # [[795, 4]]
# print(labels.shape)   # (795, 4)
# print(labels)         # [[False  True False False]  ...
labels = labels.astype(np.float64)
# print(labels)          # [[0. 1. 0. 0.]   ...

"""
We need to scale our data, because unscaled input data can result in a slow or unstable learning process. We
will use the function scale from sklearn/preprocessing . It standardizes a dataset along any axis.
It centers to the mean and component wise scale to unit variance.
"""
data = preprocessing.scale(data)
# print(data[:5])                 # ex : [[ 0.88100254 -0.30278086 -0.67388354  0.75497693] ...
# print(data.shape)                # (795, 4)
# print(labels.shape)              # (795, 4)

res = train_test_split(data, labels,
                       train_size=0.8,
                       test_size=0.2,
                       random_state=42)

train_data, test_data, train_labels, test_labels = res

"""
Use NN with bias node
"""
simple_network = NeuralNetwork(no_of_in_nodes=4,
                               no_of_out_nodes=4,
                               no_of_hidden_nodes=20,
                               learning_rate=0.3  # bias=1
                               )

for i in range(len(train_data)):
    simple_network.train(train_data[i], train_labels[i])

tr_result = simple_network.evaluate(train_data, train_labels)
print(tr_result)

te_result = simple_network.evaluate(test_data, test_labels)
print(te_result)
