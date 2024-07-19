import numpy as np
import matplotlib.pyplot as plt

x1 = np.linspace(0, 10)
x2 = np.linspace(0, 9)

y1 = np.random.rand(len(x1))
y2 = np.random.rand(len(x2))

fig1 = plt.figure()
plt.plot(x1, y1, linestyle='solid', label='first')
plt.plot(x2, y2, linestyle='dotted', label='second')
plt.grid()
plt.xlabel('X-value')
plt.ylabel('Y-value')
plt.legend(loc='best')
plt.show()