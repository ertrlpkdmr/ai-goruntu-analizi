"""
Ozel modeli degerlendirme: confusion matrix, sinif bazli metrikler,
top-2 dogruluk ve yanlis siniflanmis ornek izgarasi.

Kullanim:
  python evaluate_model.py                      # V2-Large (varsayilan)
  python evaluate_model.py --arch v2s --weights best_custom_model_v2s.pth --out evaluation_results_v2s
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--arch", choices=["v2l", "v2s"], default="v2l",
                    help="Model mimarisi (v2l=Large, v2s=Small)")
parser.add_argument("--weights", default="best_custom_model.pth",
                    help="Model agirlik dosyasi (BASE_DIR'a gore)")
parser.add_argument("--out", default="evaluation_results",
                    help="Cikti klasoru (BASE_DIR'a gore)")
args = parser.parse_args()

VAL_DIR = os.path.join(BASE_DIR, "custom_data", "val")
MODEL_PATH = os.path.join(BASE_DIR, args.weights)
CLASSES_JSON = os.path.join(BASE_DIR, "custom_classes.json")
OUT_DIR = os.path.join(BASE_DIR, args.out)
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

rcParams["font.family"] = "DejaVu Sans"

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(f"Cihaz: {DEVICE}")
print(f"Validation klasoru: {VAL_DIR}")

val_dataset = datasets.ImageFolder(VAL_DIR, transform=eval_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ImageFolder kendi alfabetik siralamasini kullanir; bunu kullanalim ki
# tahmin indeksleri verisetiyle birebir eslesssin.
DATASET_CLASSES = val_dataset.classes
print(f"Toplam ornek: {len(val_dataset)} | Sinif sayisi: {len(DATASET_CLASSES)}")

arch_name = "EfficientNet V2 Large" if args.arch == "v2l" else "EfficientNet V2 Small"
print(f"\nModel yukleniyor ({arch_name})...")
if args.arch == "v2l":
    model = models.efficientnet_v2_l(weights=None)
else:
    model = models.efficientnet_v2_s(weights=None)
in_feats = model.classifier[1].in_features
# Her iki checkpoint de Sequential(Dropout, Linear) yapisinda — classifier.1.weight var.
model.classifier[1] = nn.Linear(in_feats, NUM_CLASSES)
state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state)
model = model.to(DEVICE).eval()
print(f"Model hazir. {sum(p.numel() for p in model.parameters())/1e6:.1f}M parametre.")

all_preds = []
all_top2 = []
all_labels = []
all_probs = []
misclassified = []  # (path, true_idx, pred_idx, conf)

print("\nDegerlendirme calistiriliyor...")
with torch.no_grad():
    sample_idx = 0
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        labels_cpu = labels.numpy()
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        top2 = np.argsort(-probs, axis=1)[:, :2]
        preds = top2[:, 0]

        all_preds.extend(preds.tolist())
        all_top2.extend(top2.tolist())
        all_labels.extend(labels_cpu.tolist())
        all_probs.extend(probs.tolist())

        for i in range(len(preds)):
            if preds[i] != labels_cpu[i]:
                path, _ = val_dataset.samples[sample_idx + i]
                misclassified.append((path, int(labels_cpu[i]), int(preds[i]), float(probs[i][preds[i]])))
        sample_idx += len(preds)

all_preds = np.array(all_preds)
all_top2 = np.array(all_top2)
all_labels = np.array(all_labels)

top1_acc = (all_preds == all_labels).mean() * 100
top2_acc = np.mean([lbl in row for lbl, row in zip(all_labels, all_top2)]) * 100

print(f"\n=== GENEL SONUC ===")
print(f"Top-1 dogruluk: %{top1_acc:.2f}")
print(f"Top-2 dogruluk: %{top2_acc:.2f}")
print(f"Yanlis siniflanmis: {len(misclassified)} / {len(val_dataset)}")

# ---- Confusion matrix ----
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(DATASET_CLASSES, rotation=45, ha="right")
ax.set_yticklabels(DATASET_CLASSES)
ax.set_xlabel("Tahmin edilen sinif")
ax.set_ylabel("Gercek sinif")
ax.set_title(f"Confusion Matrix (Top-1: %{top1_acc:.2f})")

# Hucre icine sayilar
thresh = cm.max() / 2.0 if cm.max() > 0 else 1
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=10)

fig.colorbar(im, ax=ax)
fig.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nConfusion matrix kaydedildi: {cm_path}")

# ---- Sinif bazli rapor ----
report_dict = classification_report(
    all_labels, all_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=DATASET_CLASSES,
    digits=3,
    output_dict=True,
    zero_division=0,
)
report_text = classification_report(
    all_labels, all_preds,
    labels=list(range(NUM_CLASSES)),
    target_names=DATASET_CLASSES,
    digits=3,
    zero_division=0,
)
print("\n=== SINIF BAZLI RAPOR ===")
print(report_text)
with open(os.path.join(OUT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Top-1 dogruluk: %{top1_acc:.2f}\n")
    f.write(f"Top-2 dogruluk: %{top2_acc:.2f}\n\n")
    f.write(report_text)

with open(os.path.join(OUT_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
    json.dump({
        "top1_accuracy": top1_acc,
        "top2_accuracy": top2_acc,
        "per_class": report_dict,
        "class_order": DATASET_CLASSES,
    }, f, ensure_ascii=False, indent=2)

# ---- Sinif bazli F1 grafigi ----
class_f1 = [report_dict[c]["f1-score"] for c in DATASET_CLASSES]
class_prec = [report_dict[c]["precision"] for c in DATASET_CLASSES]
class_rec = [report_dict[c]["recall"] for c in DATASET_CLASSES]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(NUM_CLASSES)
w = 0.27
ax.bar(x - w, class_prec, w, label="Precision", color="#4C72B0")
ax.bar(x,     class_rec,  w, label="Recall",    color="#DD8452")
ax.bar(x + w, class_f1,   w, label="F1",        color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(DATASET_CLASSES, rotation=45, ha="right")
ax.set_ylim(0, 1.05)
ax.set_ylabel("Skor")
ax.set_title("Sinif Bazli Precision / Recall / F1")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
metrics_path = os.path.join(OUT_DIR, "per_class_metrics.png")
fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Sinif bazli metrik grafigi: {metrics_path}")

# ---- Yanlis siniflanmis ornekler ----
if misclassified:
    n_show = min(20, len(misclassified))
    # En "guvenli yanlislari" en uste koy — yani modelin emin oldugu halde hata yaptiklari
    misclassified.sort(key=lambda x: -x[3])
    show = misclassified[:n_show]
    cols = 5
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for i, (path, true_i, pred_i, conf) in enumerate(show):
        try:
            img = Image.open(path).convert("RGB")
            axes[i].imshow(img)
        except Exception as e:
            axes[i].text(0.5, 0.5, f"hata: {e}", ha="center", va="center")
        axes[i].set_title(
            f"Gercek: {DATASET_CLASSES[true_i]}\nTahmin: {DATASET_CLASSES[pred_i]} (%{conf*100:.1f})",
            fontsize=9,
        )
        axes[i].axis("off")
    for j in range(n_show, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Yanlis Siniflanmis Ornekler (en yuksek 20 / toplam {len(misclassified)})",
                 fontsize=12)
    fig.tight_layout()
    mis_path = os.path.join(OUT_DIR, "misclassified_examples.png")
    fig.savefig(mis_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Yanlis siniflanmis ornekler: {mis_path}")

    # Yanlislari CSV olarak da kaydet
    with open(os.path.join(OUT_DIR, "misclassified.csv"), "w", encoding="utf-8") as f:
        f.write("path,gercek,tahmin,guven\n")
        for path, true_i, pred_i, conf in misclassified:
            f.write(f"{path},{DATASET_CLASSES[true_i]},{DATASET_CLASSES[pred_i]},{conf:.4f}\n")

print(f"\nTum ciktilar: {OUT_DIR}")
