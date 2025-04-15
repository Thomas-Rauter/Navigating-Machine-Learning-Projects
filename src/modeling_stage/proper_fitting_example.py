import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial

# === Generate data ===
np.random.seed(0)
x = np.linspace(-3, 3, 100)
y_true = 1.0 + 0.5 * x**2
noise = np.random.normal(0, 0.5, size=x.shape)
y = y_true + noise

# === Fit a quadratic model ===
coefs = np.polyfit(x, y, deg=2)
y_pred = np.polyval(coefs, x)

# === Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', alpha=0.7, label="Data", s=50)
plt.plot(x, y_pred, color='blue', linewidth=3, label="Quadratic Fit")
plt.xlabel("x", fontsize=18)
plt.ylabel("y", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()

plt.savefig("../../figures/modeling_stage/proper_fit_example.png", dpi=300)
plt.show()
