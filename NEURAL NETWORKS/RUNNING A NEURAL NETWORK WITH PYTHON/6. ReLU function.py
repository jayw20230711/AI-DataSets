"""
ReLU - Rectified linear Unit (also called Ramp Function):
Another easy to use activation function is the ReLU function.
It is also known as the ramp function. It is defined as the positve part of its argument, i.e. y = max (0, x). This is
"currently, the most successful and widely-used activation function is the Rectified Linear Unit (ReLU)"1 The
ReLu function is computationally more efficient than Sigmoid like functions, because Relu means only
choosing the maximum between 0 and the argument x . Whereas Sigmoids need to perform expensive
exponential operations.
"""
import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0.0, x)

# derivation of relu
def ReLU_derivation(x):
    if x <= 0:
        return 0
    else:
        return 1

X = np.linspace(-5, 6, 100)
# print(X)
plt.plot(X, ReLU(X), 'b')
# plt.plot(X, ReLU_derivation(X), 'r')    # does not work
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('ReLU function')
plt.grid()
plt.text(0.8, 0.4, r'$ReLU(x) = max(0, x)$', fontsize=14)
plt.show()
