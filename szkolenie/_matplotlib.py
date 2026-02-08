import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2 * np.pi, 2 * np.pi, 100) 

y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin, "r-", label="sin")
plt.plot(x, y_cos, "b--", label="cos")

plt.ylabel("Wartość")
plt.xlabel("x")
plt.legend()
plt.show()
