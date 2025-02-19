from scipy.special import expit
import numpy as np

print(expit(3.4))                       # 0.9677045353015494
print(expit([3, 4, 1]))
print(expit(np.array([0.8, 2.3, 8])))
