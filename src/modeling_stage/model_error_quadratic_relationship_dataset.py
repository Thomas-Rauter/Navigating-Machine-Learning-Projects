import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# === Generate quadratic data ===
np.random.seed(0)
x = np.linspace(-3, 3, 100)
y_true = 1.0 + 0.5 * x**2
noise = np.random.normal(0, 0.5, size=x.shape)
y = y_true + noise

# === Fit linear model ===
X_lin = x.reshape(-1, 1)
model = LinearRegression()
model.fit(X_lin, y)
y_pred = model.predict(X_lin)

# === Plot ===
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='gray', alpha=0.7, label="Data", s=50)
plt.plot(x, y_pred, color='blue', linewidth=3, label="Linear Fit")
plt.plot(x, y_true, color='black', linestyle='--', linewidth=2, label="True (Quadratic) Relationship")
plt.xlabel("x", fontsize=18)
plt.ylabel("y", fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=16)
plt.tight_layout()

# Save the figure
output_path = "../../figures/modeling_stage/model_error_quadratic_vs_linear_fit.png"
plt.savefig(output_path, dpi=300)
plt.show()
