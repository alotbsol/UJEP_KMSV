import numpy as np
import matplotlib.pyplot as plt


heat_map = []*10
print(heat_map)
a = np.random.random((16, 16))
b = np.zeros((20, 20))
print(b)
print(a)
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()