import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
df_1mb = pd.read_csv("experiment-1mb-context/metrics.csv")
df_2k = pd.read_csv("experiment-large-context/tsinfer_2k_logs/csv_metrics/version_2/metrics.csv")

# Separate train/val rows
train_1mb = df_1mb[df_1mb["train_loss"].notna()].iloc[2:]  # skip warmup outliers
val_1mb = df_1mb[df_1mb["val_loss"].notna()]
train_2k = df_2k[df_2k["train_loss"].notna()].iloc[2:]
val_2k = df_2k[df_2k["val_loss"].notna()]

window = 20
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("1 MB Context (5k windows) vs 2k Windows — Mosquito AnoGam", fontsize=14)

# --- Train Loss ---
ax = axes[0, 0]
ax.plot(train_2k["epoch"] + train_2k["step"] / train_2k["step"].max(),
        train_2k["train_loss"].rolling(window).mean(),
        color="C0", linewidth=1.5, alpha=0.8, label="2k window")
ax.plot(train_1mb["epoch"] + train_1mb["step"] / train_1mb["step"].max(),
        train_1mb["train_loss"].rolling(window).mean(),
        color="C1", linewidth=1.5, alpha=0.8, label="1MB (5k window)")
ax.set_ylabel("Train Loss (rolling avg)")
ax.set_xlabel("Epoch")
ax.set_title("Train Loss")
ax.legend()

# --- Train RMSE ---
ax = axes[0, 1]
ax.plot(train_2k["epoch"] + train_2k["step"] / train_2k["step"].max(),
        train_2k["train_rmse"].rolling(window).mean(),
        color="C0", linewidth=1.5, alpha=0.8, label="2k window")
ax.plot(train_1mb["epoch"] + train_1mb["step"] / train_1mb["step"].max(),
        train_1mb["train_rmse"].rolling(window).mean(),
        color="C1", linewidth=1.5, alpha=0.8, label="1MB (5k window)")
ax.set_ylabel("Train RMSE (rolling avg)")
ax.set_xlabel("Epoch")
ax.set_title("Train RMSE")
ax.legend()

# --- Val Loss ---
ax = axes[1, 0]
ax.plot(val_2k["epoch"], val_2k["val_loss"], "o-", color="C0", markersize=3, label="2k window")
ax.plot(val_1mb["epoch"], val_1mb["val_loss"], "s-", color="C1", markersize=4, label="1MB (5k window)")
ax.set_ylabel("Val Loss")
ax.set_xlabel("Epoch")
ax.set_title("Validation Loss")
ax.legend()

# --- Val RMSE ---
ax = axes[1, 1]
ax.plot(val_2k["epoch"], val_2k["val_rmse"], "o-", color="C0", markersize=3, label="2k window RMSE")
ax.plot(val_1mb["epoch"], val_1mb["val_rmse"], "s-", color="C1", markersize=4, label="1MB (5k window) RMSE")
ax.set_ylabel("Val RMSE")
ax.set_xlabel("Epoch")
ax.set_title("Validation RMSE")
ax.legend()

plt.tight_layout()
plt.savefig("experiment-1mb-context/metrics_comparison.png", dpi=150)
plt.show()
print("Saved to experiment-1mb-context/metrics_comparison.png")
