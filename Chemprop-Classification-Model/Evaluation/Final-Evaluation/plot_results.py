import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

data = {
    "AChE":  {"N":70, "Active":18, "Inactive":52, "AUROC":0.9567, "MCC":0.8904, "Sensitivity":0.9444, "Specificity":0.9615, "F1_macro":0.9449, "TP":17, "FP":2, "TN":50, "FN":1},
    "BuChE": {"N":16, "Active":9,  "Inactive":7,  "AUROC":1.0000, "MCC":1.0000, "Sensitivity":1.0000, "Specificity":1.0000, "F1_macro":1.0000, "TP":9,  "FP":0, "TN":7,  "FN":0},
    "BACE1": {"N":20, "Active":12, "Inactive":8,  "AUROC":1.0000, "MCC":1.0000, "Sensitivity":1.0000, "Specificity":1.0000, "F1_macro":1.0000, "TP":12, "FP":0, "TN":8,  "FN":0},
    "MAO_B": {"N":24, "Active":11, "Inactive":13, "AUROC":0.9860, "MCC":0.9199, "Sensitivity":1.0000, "Specificity":0.9231, "F1_macro":0.9583, "TP":11, "FP":1, "TN":12, "FN":0},
}

targets = ["AChE", "BuChE", "BACE1", "MAO_B"]
metrics = ["AUROC", "MCC", "Sensitivity", "Specificity", "F1_macro"]
metric_labels = ["AUROC", "MCC", "Sensitivity", "Specificity", "F1 Macro"]

colors = ["#01696f", "#0c4e54", "#4f98a3", "#6daa45", "#d19900"]
tpr_color = "#01696f"
fpr_color = "#a12c7b"

fig, axes = plt.subplots(
    2, 4, figsize=(20, 8.5),
    gridspec_kw={"height_ratios": [1, 0.55]}
)
fig.suptitle(
    "External Validation",
    fontsize=15, fontweight="bold", y=1.02
)

x = np.arange(len(metrics))
bar_w = 0.58

for col, t in enumerate(targets):
    d = data[t]
    vals = [d[m] for m in metrics]

    ax = axes[0, col]
    bars = ax.bar(x, vals, color=colors, width=bar_w, edgecolor="white", linewidth=0.6)

    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.012,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=8.5, fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title(
        f"{t}\nN={d['N']}  Active={d['Active']}  Inactive={d['Inactive']}",
        fontsize=10, fontweight="bold"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if col == 0:
        ax.set_ylabel("Metric value", fontsize=10)

    ax2 = axes[1, col]
    tpr = d["Sensitivity"]
    fpr = d["FP"] / (d["FP"] + d["TN"]) if (d["FP"] + d["TN"]) > 0 else 0.0

    rate_bars = ax2.bar(
        [0, 1], [tpr, fpr],
        color=[tpr_color, fpr_color],
        width=0.5, edgecolor="white", linewidth=0.6
    )

    for bar, v in zip(rate_bars, [tpr, fpr]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.015,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold"
        )

    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["TPR", "FPR"], fontsize=10)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(["0", "0.5", "1.0"], fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    if col == 0:
        ax2.set_ylabel("Rate", fontsize=10)

legend_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, metric_labels)]
legend_patches += [
    mpatches.Patch(color=tpr_color, label="TPR"),
    mpatches.Patch(color=fpr_color, label="FPR"),
]

fig.legend(
    handles=legend_patches,
    loc="lower center",
    ncol=7,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.03)
)

plt.tight_layout()
plt.savefig("external_validation.png", dpi=300, bbox_inches="tight")
plt.show()
