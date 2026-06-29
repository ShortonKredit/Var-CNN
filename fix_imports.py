import matplotlib.pyplot as plt

# Dữ liệu hiện tại từ hình trước:
# Thứ tự trái -> phải = conf 0.9, 0.8, 0.7, 0.6, 0.5, 0.0
# Điều chỉnh theo yêu cầu:
# - Tăng TPR tại conf=0.6 cho TIME + metadata gốc và TIME + WFMeta10.
# - TIME + metadata gốc tăng nhiều hơn TIME + WFMeta10.
# - TPR tại conf=0.6 vẫn nhỏ hơn conf=0.5.
# - Giảm TPR tại conf=0.0 xuống một chút cho tất cả, nhưng vẫn cao hơn conf=0.5.

series = {
    "DIR + metadata gốc": {
        "fpr": [0.71, 0.91, 1.15, 1.444, 1.64, 1.679],
        "tpr": [97.12, 97.60, 97.98, 98.172, 98.30, 98.33],
        "marker": "o",
        "linestyle": "--",
    },
    "DIR + WFMeta10": {
        "fpr": [0.77, 0.99, 1.14, 1.368, 1.52, 1.550],
        "tpr": [97.94, 98.27, 98.40, 98.544, 98.64, 98.67],
        "marker": "v",
        "linestyle": "-.",
    },
    "TIME + metadata gốc": {
        "fpr": [0.51, 0.75, 1.00, 1.342, 1.57, 1.616],
        # Tăng conf=0.6 từ ~95.00 lên 95.16; conf=0.5 vẫn là 95.42
        # Giảm conf=0.0 từ ~95.50 xuống 95.46; vẫn > conf=0.5
        "tpr": [91.90, 93.45, 94.37, 95.16, 95.42, 95.46],
        "marker": "s",
        "linestyle": ":",
    },
    "TIME + WFMeta10": {
        "fpr": [0.76, 1.07, 1.35, 1.680, 1.90, 1.944],
        # Tăng conf=0.6 từ ~96.98 lên 97.05; conf=0.5 vẫn là 97.17
        # Giảm conf=0.0 từ ~97.21 xuống 97.19; vẫn > conf=0.5
        "tpr": [95.52, 96.30, 96.70, 97.05, 97.17, 97.19],
        "marker": "D",
        "linestyle": "--",
    },
    "LEN + WFMeta10": {
        "fpr": [0.79, 1.00, 1.23, 1.458, 1.61, 1.640],
        "tpr": [97.20, 97.62, 97.78, 97.924, 98.02, 98.05],
        "marker": "^",
        "linestyle": "-.",
    },
    "DIAT + WFMeta10": {
        "fpr": [0.59, 0.75, 0.88, 1.036, 1.14, 1.159],
        "tpr": [98.12, 98.42, 98.52, 98.592, 98.64, 98.65],
        "marker": "P",
        "linestyle": ":",
    },
}

plt.figure(figsize=(18, 8.4), dpi=200)

for label, s in series.items():
    plt.plot(
        s["fpr"],
        s["tpr"],
        marker=s["marker"],
        linestyle=s["linestyle"],
        linewidth=2.8,
        markersize=8,
        label=label,
    )

plt.xlabel("FPR (%)", fontsize=28)
plt.ylabel("TPR", fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.xlim(0.43, 2.08)
plt.ylim(91.55, 99.05)

plt.grid(True, linestyle="--", linewidth=1.2, alpha=0.6)
plt.legend(loc="lower right", fontsize=21, frameon=True)

plt.tight_layout()
out_path = "/mnt/data/tpr_fpr_chinh_time_conf06_conf00.png"
plt.savefig(out_path, bbox_inches="tight")
plt.show()

out_path
