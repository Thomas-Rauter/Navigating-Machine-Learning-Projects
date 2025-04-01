import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# Load sample dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Create PDP
fig, ax = plt.subplots(figsize=(6, 4))
disp = PartialDependenceDisplay.from_estimator(
    model, X_train, features=["AveRooms"], ax=ax
)

# Override axis labels manually
disp.axes_[0, 0].set_xlabel("Average Number of Rooms")
disp.axes_[0, 0].set_ylabel("Predicted Median House Value")

# Clean layout and save
plt.tight_layout()
plt.savefig("figures/pdp_avg_rooms.png", dpi=300)
