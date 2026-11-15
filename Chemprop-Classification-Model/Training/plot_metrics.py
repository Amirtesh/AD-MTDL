import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

outdir = "output/val_test_plots"
os.makedirs(outdir, exist_ok=True)

metrics = pd.DataFrame([
    ["AChE_class",  "test", 306, 0.3775, 0.9277, 0.9632, 0.7143, 0.8627, 0.8694, 0.8627, 0.8643, 0.8493, 0.8651, 0.8552, 0.9227, 0.8564, 0.8883],
    ["AChE_class",  "val",  196, 0.3775, 0.9359, 0.9404, 0.8063, 0.9031, 0.9036, 0.9031, 0.9031, 0.9027, 0.9036, 0.9029, 0.9200, 0.8932, 0.9064],
    ["BACE1_class", "test", 531, 0.4013, 0.9856, 0.9983, 0.8066, 0.9623, 0.9630, 0.9623, 0.9626, 0.8972, 0.9094, 0.9032, 0.9809, 0.9768, 0.9789],
    ["BACE1_class", "val",  554, 0.4013, 0.9837, 0.9972, 0.8441, 0.9639, 0.9635, 0.9639, 0.9637, 0.9269, 0.9173, 0.9220, 0.9771, 0.9812, 0.9792],
    ["BuChE_class", "test", 173, 0.9608, 0.9661, 0.9678, 0.8718, 0.9364, 0.9365, 0.9364, 0.9365, 0.9354, 0.9364, 0.9359, 0.9468, 0.9368, 0.9418],
    ["BuChE_class", "val",  122, 0.9608, 0.9938, 0.9941, 0.9192, 0.9590, 0.9602, 0.9590, 0.9590, 0.9598, 0.9594, 0.9590, 0.9831, 0.9355, 0.9587],
    ["MAO_B_class", "test", 187, 0.0724, 0.8620, 0.8838, 0.5493, 0.7701, 0.7785, 0.7701, 0.7693, 0.7765, 0.7727, 0.7696, 0.8293, 0.7010, 0.7598],
    ["MAO_B_class", "val",  302, 0.0724, 0.8659, 0.8381, 0.6219, 0.8146, 0.8143, 0.8146, 0.8144, 0.8115, 0.8104, 0.8109, 0.7907, 0.7786, 0.7846],
], columns=["Task","Split","N","Threshold","AUROC","AUPRC","MCC","Accuracy",
            "Prec_W","Rec_W","F1_W","Prec_Mac","Rec_Mac","F1_Mac","Prec_Pos","Rec_Pos","F1_Pos"])

cms = {
    ("AChE_class",  "test"): np.array([[97, 14], [27, 168]]),
    ("AChE_class",  "val"):  np.array([[85,  8], [11,  92]]),
    ("BACE1_class", "test"): np.array([[48,  9], [11, 463]]),
    ("BACE1_class", "val"):  np.array([[64, 11], [ 9, 470]]),
    ("BuChE_class", "test"): np.array([[71,  7], [ 3,  92]]),
    ("BuChE_class", "val"):  np.array([[58,  2], [ 3,  59]]),
    ("MAO_B_class", "test"): np.array([[74, 16], [28,  69]]),
    ("MAO_B_class", "val"):  np.array([[143,28], [29, 102]]),
}

# ── Style ──────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

TASK_LABELS = [
    "Acetylcholinesterase",
    "Beta-Secretase 1",
    "Butyrylcholinesterase",
    "Monoamine Oxidase B",
]
TASK_ORDER  = ["AChE_class", "BACE1_class", "BuChE_class", "MAO_B_class"]
PALETTE     = {"val": "#F58518", "test": "#54A24B"}

# Full names used in confusion matrix titles
TASK_FULL_NAMES = {
    "AChE_class":  "Acetylcholinesterase",
    "BACE1_class": "Beta-Secretase 1",
    "BuChE_class": "Butyrylcholinesterase",
    "MAO_B_class": "Monoamine Oxidase B",
}

# Full names for metric axis/title labels
METRIC_FULL_NAMES = {
    "AUROC":    "Area Under ROC Curve (AUROC)",
    "AUPRC":    "Area Under Precision-Recall Curve (AUPRC)",
    "MCC":      "Matthews Correlation Coefficient (MCC)",
    "Accuracy": "Accuracy",
    "F1_W":     "Weighted F1 Score",
    "Prec_Mac": "Macro Precision",
    "Rec_Mac":  "Macro Recall",
    "F1_Mac":   "Macro F1 Score",
    "Prec_Pos": "Positive Class Precision",
    "Rec_Pos":  "Positive Class Recall",
    "F1_Pos":   "Positive Class F1 Score",
}

def plot_metrics_set_1(df, metrics_list, split_order=("val", "test"), fname="metrics_set_1.png"):
    # constrained_layout ensures rotated labels don't get overlapped or cut off
    fig, axes = plt.subplots(
        2, 2, 
        figsize=(11, 9), 
        constrained_layout=True
    )
    # Minimal padding between panels while keeping labels fully legible
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, wspace=0.08, hspace=0.08)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, metrics_list):
        sns.barplot(
            data=df,
            x="Task", y=metric, hue="Split",
            order=TASK_ORDER,
            hue_order=list(split_order),
            palette=PALETTE,
            width=0.65,
            ax=ax,
        )

        ymin = df[metric].min() - 0.05
        ymin = max(0.0, round(ymin - 0.01, 2))
        ax.set_ylim(ymin, 1.04)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        ax.set_title("")  # No heading
        ax.set_xlabel("")
        ax.set_ylabel(METRIC_FULL_NAMES.get(metric, metric), fontsize=8.5)
        
        # Keep labels on all rows
        ax.set_xticks(range(len(TASK_ORDER)))
        ax.set_xticklabels(TASK_LABELS, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                fontsize=7,
                padding=3,
                rotation=90,
                label_type="edge",
            )

        leg = ax.get_legend()
        if leg:
            leg.remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(split_order),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
    )

    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {fname}")


# ── Standard Bar-metric plots for the rest ─────────────────────────────────
def plot_metrics(df, metrics_list, split_order=("val", "test"), fname="metrics.png"):
    n_metrics = len(metrics_list)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(5.4 * n_metrics, 6.0),
        constrained_layout=True,
    )
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_list):
        sns.barplot(
            data=df,
            x="Task", y=metric, hue="Split",
            order=TASK_ORDER,
            hue_order=list(split_order),
            palette=PALETTE,
            width=0.65,
            ax=ax,
        )

        ymin = df[metric].min() - 0.05
        ymin = max(0.0, round(ymin - 0.01, 2))
        ax.set_ylim(ymin, 1.04)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

        full_name = METRIC_FULL_NAMES.get(metric, metric)
        ax.set_title(full_name, fontsize=9, fontweight="bold", pad=6, wrap=True)
        ax.set_xlabel("")
        ax.set_ylabel(full_name, fontsize=8)
        ax.set_xticks(range(len(TASK_ORDER)))
        ax.set_xticklabels(TASK_LABELS, fontsize=8, rotation=40, ha="right")

        for container in ax.containers:
            ax.bar_label(
                container,
                fmt="%.3f",
                fontsize=7,
                padding=3,
                rotation=90,
                label_type="edge",
            )

        leg = ax.get_legend()
        if leg:
            leg.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=len(split_order),
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        fontsize=10,
    )

    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {fname}")


# Modified execution for set 1, original for others
plot_metrics_set_1(metrics, ["AUROC", "AUPRC", "MCC", "Accuracy"], fname="metrics_set_1.png")
plot_metrics(metrics, ["F1_W",  "Prec_Mac", "Rec_Mac", "F1_Mac"],  fname="metrics_set_2.png")
plot_metrics(metrics, ["Prec_Pos", "Rec_Pos", "F1_Pos"],           fname="metrics_pos.png")


# ── Confusion matrices — one file per target ───────────────────────────────
def plot_cm_pair(task, ax_val, ax_test, metrics_df, cms_dict):
    """Draw val and test confusion matrices for *task* onto the two axes."""
    for ax, split in zip([ax_val, ax_test], ["val", "test"]):
        cm   = cms_dict[(task, split)].astype(float)
        norm = cm / cm.sum(axis=1, keepdims=True)

        annot = np.array([
            [f"{int(cm[r, c])}\n({norm[r, c]*100:.1f}%)" for c in range(2)]
            for r in range(2)
        ])

        sns.heatmap(
            norm, annot=annot, fmt="", cmap="Blues",
            vmin=0, vmax=1,
            cbar=True,
            square=True,
            linewidths=1.5, linecolor="white",
            xticklabels=["Pred Neg", "Pred Pos"],
            yticklabels=["Actual Neg", "Actual Pos"],
            ax=ax,
            annot_kws={"fontsize": 11, "fontweight": "bold"},
        )

        row   = metrics_df[(metrics_df.Task == task) & (metrics_df.Split == split)].iloc[0]
        full_task = TASK_FULL_NAMES[task]
        split_label = "Validation" if split == "val" else "Test"
        ax.set_title(
            f"{full_task}  —  {split_label} Set  "
            f"(N={int(row.N)}, threshold={row.Threshold:.2f})",
            fontsize=11, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Predicted label", fontsize=10)
        ax.set_ylabel("True label",      fontsize=10)
        ax.tick_params(labelsize=10)


for task in TASK_ORDER:
    label = task.replace("_class", "")
    fig, (ax_val, ax_test) = plt.subplots(
        1, 2,
        figsize=(11, 5),
        constrained_layout=True,
    )
    plot_cm_pair(task, ax_val, ax_test, metrics, cms)

    fname = f"cm_{label}.png"
    fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {fname}")


# ── CSV ────────────────────────────────────────────────────────────────────
metrics.to_csv(os.path.join(outdir, "val_test_metrics.csv"), index=False)
print("Done. All outputs written to:", outdir)
