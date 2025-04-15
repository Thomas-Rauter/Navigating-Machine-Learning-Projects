# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
# from sklearn.neighbors import KernelDensity
# from PIL import Image
#
# # === Parameters ===
# np.random.seed(42)
# n_points = 50  # Reduced number of points
# means = [0, 0]
# stds = [0.3, 0.6, 1.1]  # professional, amateur, beginner
#
# # === Generate throws ===
# pro_throws = np.random.normal(loc=means, scale=stds[0], size=(n_points, 2))
# am_throws = np.random.normal(loc=means, scale=stds[1], size=(n_points, 2))
# beg_throws = np.random.normal(loc=means, scale=stds[2], size=(n_points, 2))
#
# # === Grid for KDE ===
# x = np.linspace(-3, 3, 300)
# y = np.linspace(-3, 3, 300)
# xx, yy = np.meshgrid(x, y)
# grid = np.vstack([xx.ravel(), yy.ravel()]).T
#
# # === KDE Estimation ===
# kde_pro = KernelDensity(bandwidth=0.5).fit(pro_throws)
# kde_am = KernelDensity(bandwidth=0.5).fit(am_throws)
# kde_beg = KernelDensity(bandwidth=0.5).fit(beg_throws)
#
# log_dens_pro = kde_pro.score_samples(grid)
# log_dens_am = kde_am.score_samples(grid)
# log_dens_beg = kde_beg.score_samples(grid)
#
# # === Predict class for each grid point ===
# all_log_dens = np.vstack([log_dens_beg, log_dens_am, log_dens_pro])
# preds = np.argmax(all_log_dens, axis=0).reshape(xx.shape)
#
# # === Plot Setup ===
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#
# # === Left: Dartboard with throws ===
# dartboard_path = '../../figures/modeling_stage/dart_board.png'
# try:
#     dartboard_img = Image.open(dartboard_path).convert("L").convert("RGBA")
#     axs[0].imshow(dartboard_img, extent=[-3, 3, -3, 3], zorder=0)
# except FileNotFoundError:
#     print("Dartboard image not found. Skipping background.")
#
# axs[0].scatter(beg_throws[:, 0], beg_throws[:, 1], color='orange', label='Beginner', s=80)
# axs[0].scatter(am_throws[:, 0], am_throws[:, 1], color='blue', label='Amateur', s=80)
# axs[0].scatter(pro_throws[:, 0], pro_throws[:, 1], color='crimson', label='Professional', s=80)
# axs[0].set_xlim(-3, 3)
# axs[0].set_ylim(-3, 3)
# axs[0].set_aspect('equal')
# axs[0].axis('off')
# axs[0].legend(
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.1),
#     ncol=3,
#     frameon=False,
#     fontsize=16
# )
#
# # === Right: Decision boundaries ===
# cmap = ListedColormap(['orange', 'blue', 'crimson'])
# axs[1].contourf(xx, yy, preds, cmap=cmap, alpha=0.7)
# axs[1].set_xlim(-3, 3)
# axs[1].set_ylim(-3, 3)
# axs[1].set_aspect('equal')
# axs[1].axis('off')
#
# plt.tight_layout()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KernelDensity
from PIL import Image

# === Parameters ===
np.random.seed(42)
n_points = 50
means = [0, 0]
stds = [0.3, 0.6, 1.1]  # professional, amateur, beginner

# === Generate throws ===
pro_throws = np.random.normal(loc=means, scale=stds[0], size=(n_points, 2))
am_throws = np.random.normal(loc=means, scale=stds[1], size=(n_points, 2))
beg_throws = np.random.normal(loc=means, scale=stds[2], size=(n_points, 2))

# === Grid for KDE ===
x = np.linspace(-3, 3, 300)
y = np.linspace(-3, 3, 300)
xx, yy = np.meshgrid(x, y)
grid = np.vstack([xx.ravel(), yy.ravel()]).T

bandwidth = 0.2  # instead of 0.5
kde_pro = KernelDensity(bandwidth=bandwidth).fit(pro_throws)
kde_am = KernelDensity(bandwidth=bandwidth).fit(am_throws)
kde_beg = KernelDensity(bandwidth=bandwidth).fit(beg_throws)

log_dens_pro = kde_pro.score_samples(grid)
log_dens_am = kde_am.score_samples(grid)
log_dens_beg = kde_beg.score_samples(grid)

# === Predict class for each grid point ===
all_log_dens = np.vstack([log_dens_beg, log_dens_am, log_dens_pro])
preds = np.argmax(all_log_dens, axis=0).reshape(xx.shape)

# === Plot: Dartboard with throws and overlaid decision boundaries ===
fig, ax = plt.subplots(figsize=(6, 6))

dartboard_path = '../../figures/modeling_stage/dart_board.png'
try:
    dartboard_img = Image.open(dartboard_path).convert("L").convert("RGBA")
    ax.imshow(dartboard_img, extent=[-3, 3, -3, 3], zorder=0)
except FileNotFoundError:
    print("Dartboard image not found. Skipping background.")

# Overlay decision boundaries with 50% transparency
cmap = ListedColormap(['orange', 'blue', 'crimson'])
ax.contourf(xx, yy, preds, cmap=cmap, alpha=0.5, zorder=1)

# Plot scatter points
ax.scatter(beg_throws[:, 0], beg_throws[:, 1], color='orange', label='Beginner', s=80, zorder=2)
ax.scatter(am_throws[:, 0], am_throws[:, 1], color='blue', label='Amateur', s=80, zorder=2)
ax.scatter(pro_throws[:, 0], pro_throws[:, 1], color='crimson', label='Professional', s=80, zorder=2)

# Layout
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.axis('off')
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, 1.08),
    ncol=3,
    frameon=False,
    fontsize=20
)

plt.tight_layout()
plt.show()
