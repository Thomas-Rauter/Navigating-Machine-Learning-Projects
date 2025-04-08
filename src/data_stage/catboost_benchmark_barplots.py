import matplotlib.pyplot as plt
import pandas as pd

# Define the benchmark data manually
data = {
    "Dataset": [
        "Adult", "Amazon", "Click prediction", "KDD appetency", "KDD churn",
        "KDD internet", "KDD upselling", "KDD 98", "Kick prediction"
    ],
    "CatBoost_Tuned": [0.26974, 0.13772, 0.39090, 0.07151, 0.23129, 0.20875, 0.16613, 0.19467, 0.28479],
    "CatBoost_Default": [0.27298, 0.13811, 0.39112, 0.07138, 0.23193, 0.22021, 0.16674, 0.19479, 0.28491],
    "LightGBM_Tuned": [0.27602, 0.16360, 0.39633, 0.07179, 0.23205, 0.22315, 0.16682, 0.19576, 0.29566],
    "LightGBM_Default": [0.28716, 0.16716, 0.39749, 0.07482, 0.23565, 0.23627, 0.17107, 0.19837, 0.29877],
    "XGBoost_Tuned": [0.27542, 0.16327, 0.39624, 0.07176, 0.23312, 0.22532, 0.16632, 0.19568, 0.29465],
    "XGBoost_Default": [0.28009, 0.16536, 0.39764, 0.07646, 0.23369, 0.23468, 0.16873, 0.19795, 0.29846],
    "H2O_Tuned": [0.27510, 0.16264, 0.39759, 0.07246, 0.23275, 0.22209, 0.16824, 0.19607, 0.29481],
    "H2O_Default": [0.27607, 0.16950, 0.39785, 0.07355, 0.23287, 0.24023, 0.16981, 0.19647, 0.29635]
}

df = pd.DataFrame(data)

# Reshape the data for plotting
df_melted = df.melt(id_vars="Dataset", var_name="Method", value_name="LogLoss")
df_melted["Model"] = df_melted["Method"].apply(lambda x: x.split("_")[0])
df_melted["Tuning"] = df_melted["Method"].apply(lambda x: x.split("_")[1])

# Setup grid layout
datasets = df["Dataset"]
models = ["CatBoost", "LightGBM", "XGBoost", "H2O"]
colors = {"Tuned": "blue", "Default": "red"}

fig, axes = plt.subplots(
    nrows=len(datasets), ncols=len(models),
    figsize=(6.5, 7), sharex=True, sharey=True
)

bar_width = 0.05  # thin bars

for i, dataset in enumerate(datasets):
    for j, model in enumerate(models):
        ax = axes[i, j]
        subset = df_melted[(df_melted["Dataset"] == dataset) & (df_melted["Model"] == model)]

        for k, (idx, row) in enumerate(subset.iterrows()):
            x_pos = k * (bar_width + 0.02)
            ax.bar(
                x=x_pos,
                height=row["LogLoss"],
                width=bar_width,
                color=colors[row["Tuning"]],
            )
            ax.text(
                x_pos, row["LogLoss"] + 0.001,
                f"{row['LogLoss']:.5f}",
                ha='center', va='bottom', fontsize=6  # Horizontal text
            )

        if i == 0:
            ax.set_title(model, fontsize=9)
        # if j == 0:
        #     ax.set_ylabel(dataset, fontsize=8)
        if j == 0:
            ax.text(
                -0.3,  # x-position (tweak if needed)
                0.5,  # y-position (centered in axis)
                dataset,
                fontsize=7,
                ha='right',
                va='center',
                rotation=0,
                transform=ax.transAxes
            )
        ax.set_xticks([])

        # Handle low-range dataset
        if dataset == "KDD appetency":
            ax.set_ylim(0.07, 0.075)
        else:
            ax.set_ylim(df_melted["LogLoss"].min() * 0.98, df_melted["LogLoss"].max() * 1.02)


# Legend just above top row, centered, inside the figure canvas
handles = [
    plt.Line2D([0], [0], color="blue", lw=4, label="Tuned"),
    plt.Line2D([0], [0], color="red", lw=4, label="Default")
]

# Add legend BEFORE tight_layout — this is key
fig.legend(
    handles=handles,
    loc="upper center",
    ncol=2,
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.59, 0.995)  # tightly above subplots
)

# Now lock layout tightly below legend
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 0.96 = just below bbox_to_anchor
plt.savefig("../../figures/data_stage/catboost_benchmark_barplots.png", dpi=300)
plt.close()



