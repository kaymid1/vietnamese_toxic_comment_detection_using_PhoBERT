import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

metrics = ["Macro-F1", "Toxic F1", "Precision", "Recall"]

tfidf = [0.7009, 0.4781, 0.4196, 0.5556]
phobert_v1 = [0.7133, 0.4980, 0.4453, 0.5648]
phobert_v2 = [0.7380, 0.5410, 0.4853, 0.6111]

x = np.arange(len(metrics))
width = 0.24

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
})

fig, ax = plt.subplots(figsize=(8.2, 4.8))

bars1 = ax.bar(
    x - width,
    tfidf,
    width,
    label="TF-IDF LR",
    color="#b8b8b8",
    edgecolor="black",
    hatch="//",
)

bars2 = ax.bar(
    x,
    phobert_v1,
    width,
    label="PhoBERT v1",
    color="#7fa6d8",
    edgecolor="black",
    hatch="..",
)

bars3 = ax.bar(
    x + width,
    phobert_v2,
    width,
    label="PhoBERT v2",
    color="#2855b6",
    edgecolor="black",
)

for bars in (bars1, bars2, bars3):
    ax.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=8,
    )

ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.0)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(loc="upper left", ncol=3, frameon=False)

fig.tight_layout()
fig.savefig(
    "figures/toxicity_model_comparison.pdf",
    bbox_inches="tight",
)
fig.savefig(
    "figures/toxicity_model_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)