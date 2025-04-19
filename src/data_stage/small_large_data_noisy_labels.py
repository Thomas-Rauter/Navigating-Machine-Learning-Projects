import numpy as np
import matplotlib.pyplot as plt

# Settings for reproducibility
np.random.seed(42)

# Generate true underlying function
def true_function(x):
    return np.log(x + 1) * 5

# Generate x values
x_small = np.linspace(0, 5, 7)
x_big = np.linspace(0, 5, 30)

# Generate noisy observations
noise_small = np.random.normal(0, 1.5, size=x_small.shape)
noise_big = np.random.normal(0, 1.5, size=x_big.shape)
noise_small_clean = np.random.normal(0, 0.2, size=x_small.shape)

y_small_noisy = true_function(x_small) + noise_small
y_big_noisy = true_function(x_big) + noise_big
y_small_clean = true_function(x_small) + noise_small_clean

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(16, 5))  # Wider for A4

# Left: Small data + noisy labels
axs[0].scatter(x_small, y_small_noisy, color='red', s=80)
for _ in range(4):  # Multiple possible fits
    random_noise = np.random.normal(0, 0.5, size=x_small.shape)
    coeffs = np.polyfit(x_small, y_small_noisy + random_noise, 2)
    poly = np.poly1d(coeffs)
    axs[0].plot(np.linspace(0,5,100), poly(np.linspace(0,5,100)), color='blue', alpha=0.7)
axs[0].set_title('Small data\nNoisy labels', fontsize=14)
axs[0].set_xlabel('X', fontsize=14)
axs[0].set_ylabel('Y', fontsize=14)
axs[0].tick_params(labelsize=12)
axs[0].set_xlim(-0.5, 5.5)
axs[0].set_ylim(-5, 15)

# Middle: Big data + noisy labels
axs[1].scatter(x_big, y_big_noisy, color='red', s=80)
coeffs_big = np.polyfit(x_big, y_big_noisy, 2)
poly_big = np.poly1d(coeffs_big)
axs[1].plot(np.linspace(0,5,100), poly_big(np.linspace(0,5,100)), color='blue', linewidth=3)
axs[1].set_title('Big data\nNoisy labels', fontsize=14)
axs[1].set_xlabel('X', fontsize=14)
axs[1].set_ylabel('Y', fontsize=14)
axs[1].tick_params(labelsize=12)
axs[1].set_xlim(-0.5, 5.5)
axs[1].set_ylim(-5, 15)

# Right: Small data + clean labels
axs[2].scatter(x_small, y_small_clean, color='red', s=80)
coeffs_clean = np.polyfit(x_small, y_small_clean, 2)
poly_clean = np.poly1d(coeffs_clean)
axs[2].plot(np.linspace(0,5,100), poly_clean(np.linspace(0,5,100)), color='blue', linewidth=3)
axs[2].set_title('Small data\nClean labels', fontsize=14)
axs[2].set_xlabel('X', fontsize=14)
axs[2].set_ylabel('Y', fontsize=14)
axs[2].tick_params(labelsize=12)
axs[2].set_xlim(-0.5, 5.5)
axs[2].set_ylim(-5, 15)

# Final layout tweaks
plt.tight_layout()
plt.savefig("../../figures/data_stage/label_quality_and_data_quantity.png",
            dpi=300)
plt.show()
