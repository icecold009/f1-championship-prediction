import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import os

df = pd.read_csv("Results/2023_predictions.csv")

fig, ax = plt.subplots(figsize=(12, 8))

x = range(len(df))
bar_width = 0.35

actual_bars = ax.barh(
    [i + bar_width/2 for i in x],
    df["Actual Position"],
    height=bar_width,
    label="Actual Position",
    color="#e10600",
    alpha=0.85
)
predicted_bars = ax.barh(
    [i - bar_width/2 for i in x],
    df["Predicted Rank"],
    height=bar_width,
    label="Predicted Rank",
    color="#1e41ff",
    alpha=0.85
)

ax.set_yticks(list(x))
ax.set_yticklabels(df["Driver"], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("Championship Position (lower = better)")
ax.set_title("2023 F1 Championship — Predicted vs Actual Standings", fontsize=13, fontweight='bold')
ax.legend()
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.tight_layout()
os.makedirs("Results", exist_ok=True)
plt.savefig("Results/predicted_vs_actual.png", dpi=150)
print("Chart saved to Results/predicted_vs_actual.png")