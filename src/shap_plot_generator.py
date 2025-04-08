import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from PIL import Image

# Load dataset
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Fit model
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42, verbosity=0)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Generate and save individual SHAP plots
shap.plots.beeswarm(shap_values, show=False)
plt.tight_layout()
plt.savefig("figures/shap_summary_tmp.png", dpi=300)
plt.close()

shap.plots.scatter(shap_values[:, "AveRooms"], color=shap_values, show=False)
plt.xlabel("Average Number of Rooms")
plt.ylabel("SHAP Value (Impact on Prediction)")
plt.title("SHAP Dependence Plot", fontsize=12)
plt.tight_layout()
plt.savefig("figures/shap_dependence_tmp.png", dpi=300)
plt.close()

# Combine the two images side by side
img1 = Image.open("figures/shap_summary_tmp.png")
img2 = Image.open("figures/shap_dependence_tmp.png")

combined_width = img1.width + img2.width
max_height = max(img1.height, img2.height)

combined = Image.new("RGB", (combined_width, max_height), (255, 255, 255))
combined.paste(img1, (0, 0))
combined.paste(img2, (img1.width, 0))
combined.save("figures/shap_summary_dependence.png")
