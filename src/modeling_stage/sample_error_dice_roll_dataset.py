import numpy as np
import matplotlib.pyplot as plt
import os

# Set seed for reproducibility
np.random.seed(42)

# Simulate dice rolls
n1, n2 = 100, 1000
rolls_100 = np.random.randint(1, 7, size=n1)
rolls_1000 = np.random.randint(1, 7, size=n2)

# Compute relative frequencies
bins = np.arange(1, 8)
freqs_100 = np.histogram(rolls_100, bins=bins)[0] / n1
freqs_1000 = np.histogram(rolls_1000, bins=bins)[0] / n2

# Determine upper y-limit (slightly above max for both)
y_max = max(freqs_100.max(), freqs_1000.max()) * 1.1

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

axs[0].bar(np.arange(1, 7), freqs_100, color='gray')
axs[0].set_title('100 Rolls', fontsize=16)
axs[0].set_xticks(np.arange(1, 7))
axs[0].set_xlabel('Dice Face', fontsize=14)
axs[0].set_ylabel('Relative Frequency', fontsize=14)
axs[0].tick_params(axis='both', labelsize=12)
axs[0].set_ylim(0, y_max)

axs[1].bar(np.arange(1, 7), freqs_1000, color='gray')
axs[1].set_title('1000 Rolls', fontsize=16)
axs[1].set_xticks(np.arange(1, 7))
axs[1].set_xlabel('Dice Face', fontsize=14)
axs[1].tick_params(axis='both', labelsize=12)

plt.tight_layout()

# Save
output_path = '../../figures/modeling_stage/sample_error_dice_roll_dataset.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()
