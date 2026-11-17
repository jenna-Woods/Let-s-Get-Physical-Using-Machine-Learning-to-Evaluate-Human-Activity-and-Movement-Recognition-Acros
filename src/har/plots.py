from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_results(root):
    root = Path(root)
    metrics = pd.read_csv(root / "metrics.csv")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    if "protocol" in metrics:
        fixed = metrics[metrics.choice.str.startswith("k_") & (metrics.model != "dummy")]
        for metric, label in [("macro_f1", "Macro-F1"), ("fit_seconds", "Pipeline fit (seconds)"),
                              ("prediction_us_per_row", "Prediction (microseconds / window)")]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for (protocol, model), group in fixed.groupby(["protocol", "model"]):
                summary = group.groupby("k")[metric].agg(["mean", "std"]).sort_index()
                ax.errorbar(summary.index, summary["mean"], yerr=summary["std"].fillna(0),
                            label=f"{model}: {protocol}", marker="o", capsize=3)
            ax.set(xlabel="Selected features", ylabel=label)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / f"features_{metric}.png", dpi=160)
            plt.close(fig)
        best = metrics[metrics.choice == "best"]
        summary = best.pivot_table(index="model", columns="protocol", values="macro_f1", aggfunc="mean")
        ax = summary.plot.bar(rot=0, ylim=(0, 1), ylabel="Mean outer-fold macro-F1", figsize=(7, 4))
        ax.figure.tight_layout()
        ax.figure.savefig(output / "protocol_comparison.png", dpi=160)
        plt.close(ax.figure)
    # Render confusion matrices, per-person results, and selection frequencies for each model.
    for path in root.rglob("confusion_normalized.csv"):
        table = pd.read_csv(path, index_col=0)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(table.to_numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(table)), table.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(table)), table.index, fontsize=8)
        ax.set(xlabel="Predicted", ylabel="True", title=str(path.parent.relative_to(root)))
        for i in range(len(table)):
            for j in range(len(table)):
                ax.text(j, i, f"{table.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if table.iloc[i, j] > .5 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(path.parent / "confusion.png", dpi=140)
        plt.close(fig)
    for path in root.rglob("per_subject.csv"):
        table = pd.read_csv(path)
        ax = table.plot.bar(x="subject", y="macro_f1", ylim=(0, 1), legend=False,
                            ylabel="Macro-F1", title="Held-out participant performance", figsize=(7, 4))
        ax.figure.tight_layout()
        ax.figure.savefig(path.parent / "per_subject.png", dpi=140)
        plt.close(ax.figure)
    if "protocol" in metrics:
        for (protocol, model), group in metrics[metrics.choice == "best"].groupby(["protocol", "model"]):
            if model == "dummy":
                continue
            tables = [pd.read_csv(root / protocol / f"fold_{row.fold}" / model /
                                  f"candidate_{row.candidate}" / "features.csv")
                      for row in group.itertuples()]
            table = tables[0][["feature_id", "name"]].copy()
            table["selection_frequency"] = sum(t.selected.astype(float) for t in tables) / len(tables)
            table.to_csv(output / f"{protocol}_{model}_feature_frequency.csv", index=False)
            # Name prefixes retain distinctions between acceleration, gyro, time/frequency, etc.
            table["family"] = table.name.str.split("-", n=1).str[0]
            table.groupby("family").selection_frequency.agg(["mean", "sum", "count"]).to_csv(
                output / f"{protocol}_{model}_feature_families.csv")
