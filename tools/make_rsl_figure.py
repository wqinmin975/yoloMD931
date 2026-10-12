"""Generate Figure 1 for the RSL letter: cross-domain localization degradation."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = r"D:/ace/source/RSL_fig1_domain_shift.png"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

# Panel (a): mean best-IoU
doms = ["VisDrone\n(in-domain)", "UAVDT-M\n(zero-shot)"]
iou = [0.40, 0.27]
bars = ax1.bar(doms, iou, color=["#2c7fb8", "#d95f0e"], width=0.55)
ax1.set_ylim(0, 0.5)
ax1.set_ylabel("Mean best per-image IoU")
ax1.set_title("(a) Localization quality", fontsize=10)
for b, v in zip(bars, iou):
    ax1.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
             ha="center", va="bottom", fontsize=9)

# Panel (b): overlap-rate degradation
metrics = [">=0.25 IoU", ">=0.50 IoU"]
vis = [79.0, 29.0]
uav = [49.7, 15.0]
x = range(len(metrics))
w = 0.35
ax2.bar([i - w/2 for i in x], vis, w, label="VisDrone", color="#2c7fb8")
ax2.bar([i + w/2 for i in x], uav, w, label="UAVDT-M", color="#d95f0e")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Images with overlap (%)")
ax2.set_title("(b) Overlap-rate degradation", fontsize=10)
ax2.set_xticks(list(x))
ax2.set_xticklabels(metrics, fontsize=9)
ax2.legend(fontsize=8)
for i, v in enumerate(vis):
    ax2.text(i - w/2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=8)
for i, v in enumerate(uav):
    ax2.text(i + w/2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("Saved figure ->", OUT)
