import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from PIL import Image

# Load and split data
X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train model
model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)

# SHAP explainer
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)

# Pick a single instance
i = 0
single_shap = shap_values[i]

# Save force plot
shap.plots.force(single_shap, matplotlib=True, show=False)
plt.tight_layout()
plt.savefig("../figures/shap_force_tmp.png", dpi=300)
plt.close()

# Save waterfall plot
shap.plots.waterfall(single_shap, show=False)
plt.tight_layout()
plt.savefig("../figures/shap_waterfall_tmp.png", dpi=300)
plt.close()

# Combine into one image side by side
img1 = Image.open("../figures/shap_force_tmp.png")
img2 = Image.open("../figures/shap_waterfall_tmp.png")

combined_width = max(img1.width, img2.width)
combined_height = img1.height + img2.height

combined = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
combined.paste(img1, (0, 0))
combined.paste(img2, (0, img1.height))
combined.save("../figures/shap_force_waterfall.png")
