import pandas as pd
import matplotlib.pyplot as plt

csv_path = r".\PHOSCnet_temporalpooling\log.csv"  
df = pd.read_csv(csv_path)

# Coerce types and drop NAs
df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
df["acc_num"] = pd.to_numeric(df["acc"], errors="coerce")
df["loss_num"] = pd.to_numeric(df["loss"], errors="coerce")

# 1) Validation Accuracy
acc_df = df.dropna(subset=["epoch", "acc_num"])
plt.figure()
plt.plot(acc_df["epoch"], acc_df["acc_num"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy vs Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("val_accuracy.png", dpi=180)

# 2) Training Loss
loss_df = df.dropna(subset=["epoch", "loss_num"])
plt.figure()
plt.plot(loss_df["epoch"], loss_df["loss_num"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=180)
