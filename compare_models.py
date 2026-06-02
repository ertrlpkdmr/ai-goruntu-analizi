"""
V2-Large vs V2-Small karsilastirma raporu ve gorseli.
Iki modelin evaluation_results klasorlerini okur, yan yana sunar.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

V2L_REPORT = os.path.join(BASE_DIR, "evaluation_results", "classification_report.json")
V2S_REPORT = os.path.join(BASE_DIR, "evaluation_results_v2s", "classification_report.json")
OUT_PATH = os.path.join(BASE_DIR, "model_comparison.png")

with open(V2L_REPORT, encoding="utf-8") as f:
    v2l = json.load(f)
with open(V2S_REPORT, encoding="utf-8") as f:
    v2s = json.load(f)

classes = v2l["class_order"]
assert classes == v2s["class_order"], "Sinif siralamasi uyusmuyor"

v2l_f1 = [v2l["per_class"][c]["f1-score"] for c in classes]
v2s_f1 = [v2s["per_class"][c]["f1-score"] for c in classes]

# ---- Karsilastirma gorseli ----
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

# Ust sol: Top-1 ve Top-2 karsilastirma
ax1 = fig.add_subplot(gs[0, 0])
labels = ["Top-1 Dogruluk", "Top-2 Dogruluk"]
v2l_vals = [v2l["top1_accuracy"], v2l["top2_accuracy"]]
v2s_vals = [v2s["top1_accuracy"], v2s["top2_accuracy"]]
x = np.arange(len(labels))
w = 0.35
b1 = ax1.bar(x - w/2, v2l_vals, w, label="V2-Large (117M)", color="#3498db")
b2 = ax1.bar(x + w/2, v2s_vals, w, label="V2-Small (20.2M)", color="#e67e22")
ax1.set_ylabel("Dogruluk (%)")
ax1.set_title("Genel Dogruluk Karsilastirmasi")
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylim(0, 105)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1, f"%{h:.2f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

# Ust sag: Model bilgisi tablosu
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
table_data = [
    ["Mimari", "EfficientNet V2-L", "EfficientNet V2-S"],
    ["Parametre", "117M", "20.2M"],
    ["Model boyutu", "~471 MB", "~81 MB"],
    ["Top-1", f"%{v2l['top1_accuracy']:.2f}", f"%{v2s['top1_accuracy']:.2f}"],
    ["Top-2", f"%{v2l['top2_accuracy']:.2f}", f"%{v2s['top2_accuracy']:.2f}"],
    ["Macro F1", f"{v2l['per_class']['macro avg']['f1-score']:.3f}",
                f"{v2s['per_class']['macro avg']['f1-score']:.3f}"],
]
table = ax2.table(cellText=table_data, colWidths=[0.35, 0.32, 0.32],
                  cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.8)
# Baslik satirini boya
for i in range(3):
    cell = table[0, i]
    cell.set_facecolor("#2c3e50")
    cell.set_text_props(color="white", fontweight="bold")
ax2.set_title("Model Karsilastirma Tablosu", pad=20)

# Alt: Sinif bazli F1 karsilastirma
ax3 = fig.add_subplot(gs[1, :])
xc = np.arange(len(classes))
w = 0.4
b1 = ax3.bar(xc - w/2, v2l_f1, w, label="V2-Large", color="#3498db")
b2 = ax3.bar(xc + w/2, v2s_f1, w, label="V2-Small", color="#e67e22")
ax3.set_xticks(xc)
ax3.set_xticklabels(classes, rotation=30, ha="right")
ax3.set_ylabel("F1-Score")
ax3.set_ylim(0, 1.1)
ax3.set_title("Sinif Bazli F1 Karsilastirmasi")
ax3.legend()
ax3.grid(axis="y", alpha=0.3)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                 ha="center", va="bottom", fontsize=8)

fig.suptitle("EfficientNet V2-Large vs V2-Small — Karsilastirma Raporu",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print(f"Karsilastirma gorseli kaydedildi: {OUT_PATH}")
print("\n=== OZET ===")
print(f"V2-Large:  Top-1 %{v2l['top1_accuracy']:.2f}  |  Top-2 %{v2l['top2_accuracy']:.2f}  |  117M param  |  471 MB")
print(f"V2-Small:  Top-1 %{v2s['top1_accuracy']:.2f}  |  Top-2 %{v2s['top2_accuracy']:.2f}  |  20.2M param  |  81 MB")
print(f"Fark:      Top-1 {v2s['top1_accuracy']-v2l['top1_accuracy']:+.2f} puan  |  Boyut 5.8x kucuk")
