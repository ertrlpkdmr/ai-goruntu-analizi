"""
Ozel Nesne Tanima - Transfer Learning ile EfficientNet V2 SMALL
Makine Ogrenmesi Dersi Projesi - V2-Large ile karsilastirma icin

Degisiklikler (train_custom.py'ye gore):
  - Model: V2-Large (117M) -> V2-Small (~21M)  -- kucuk veri seti icin daha uygun
  - Epoch: 30 -> 50  -- daha uzun egitim
  - Early stopping: val 12 epoch iyilesmezse durur
  - Ciktilar ayri: best_custom_model_v2s.pth vs eski model korunur
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import json

# =============================================
# 1. AYARLAR
# =============================================
BATCH_SIZE = 32
EPOCHS = 50
FREEZE_EPOCHS = 5
LR_CLASSIFIER = 1e-3
LR_BACKBONE = 1e-5
NUM_WORKERS = 2
EARLY_STOP_PATIENCE = 12  # fine-tune fazinda iyileşme yoksa erken durdur
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "custom_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_custom_model_v2s.pth")
TRAINING_PLOT = os.path.join(BASE_DIR, "custom_training_results_v2s.png")
CLASS_ACC_PLOT = os.path.join(BASE_DIR, "custom_class_accuracy_v2s.png")

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5

print(f"Kullanilan cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================
# 2. VERI HAZIRLAMA
# =============================================
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("\nVeri seti yukleniyor...")
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)

CLASSES = train_dataset.classes
NUM_CLASSES = len(CLASSES)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"Egitim verisi: {len(train_dataset)} gorsel")
print(f"Validation verisi: {len(val_dataset)} gorsel")
print(f"Sinif sayisi: {NUM_CLASSES}")
print(f"Siniflar: {CLASSES}")

# =============================================
# 3. MIXUP VE CUTMIX
# =============================================
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    _, _, H, W = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================
# 4. MODEL (EfficientNet V2 SMALL)
# =============================================
print("\nModel yukleniyor (EfficientNet V2 Small - Transfer Learning)...")
model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES)
)

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Toplam parametre: {total_params:,} (~{total_params/1e6:.1f}M)")
print(f"Egitilebilir parametre: {trainable_params:,}")

# =============================================
# 5. KAYIP VE OPTIMIZER
# =============================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone donduruldu. Egitilebilir parametre: {trainable:,}")


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Tum model acildi. Egitilebilir parametre: {trainable:,}")


# =============================================
# 6. EGITIM/TEST FONKSIYONLARI
# =============================================
def train_one_epoch(model, loader, criterion, optimizer, device, use_mix=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        if use_mix and inputs.size(0) > 1:
            if np.random.random() < MIXUP_PROB:
                inputs, ta, tb, lam = mixup_data(inputs, targets, MIXUP_ALPHA)
            else:
                inputs, ta, tb, lam = cutmix_data(inputs, targets, CUTMIX_ALPHA)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, ta, tb, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(ta).sum().float()
                        + (1 - lam) * predicted.eq(tb).sum().float()).item()
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100.0 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = 100.0 * correct / total if total > 0 else 0
    avg_loss = running_loss / len(loader) if len(loader) > 0 else 0
    return avg_loss, accuracy


# =============================================
# 7. EGITIM DONGUSU
# =============================================
print("\n" + "=" * 60)
print("EGITIM BASLIYOR")
print(f"Model: EfficientNet V2 Small | Epoch: {EPOCHS} | Batch: {BATCH_SIZE}")
print(f"Freeze Epochs: {FREEZE_EPOCHS} | LR clf: {LR_CLASSIFIER} | LR backbone: {LR_BACKBONE}")
print(f"Mixup: alpha={MIXUP_ALPHA} | CutMix: alpha={CUTMIX_ALPHA}")
print(f"Early stopping patience: {EARLY_STOP_PATIENCE} epoch")
print("=" * 60)

train_losses, train_accs = [], []
val_losses, val_accs = [], []
best_acc = 0.0
best_epoch = 0
epochs_without_improvement = 0
stopped_early = False

start_time = time.time()

# --- FAZE 1 ---
print(f"\n--- FAZE 1: Classifier egitimi (epoch 1-{FREEZE_EPOCHS}) ---")
freeze_backbone(model)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_CLASSIFIER, weight_decay=1e-4
)
scheduler = CosineAnnealingLR(optimizer, T_max=FREEZE_EPOCHS, eta_min=1e-5)

for epoch in range(1, FREEZE_EPOCHS + 1):
    epoch_start = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, use_mix=False
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()
    train_losses.append(train_loss); train_accs.append(train_acc)
    val_losses.append(val_loss); val_accs.append(val_acc)

    epoch_time = time.time() - epoch_start
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        marker = " * En iyi!"
    else:
        marker = ""
    print(
        f"Epoch [{epoch:3d}/{EPOCHS}] | "
        f"Egitim: {train_loss:.4f} / %{train_acc:.2f} | "
        f"Val: {val_loss:.4f} / %{val_acc:.2f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
        f"{epoch_time:.1f}s{marker}"
    )

# --- FAZE 2 ---
remaining_epochs = EPOCHS - FREEZE_EPOCHS
print(f"\n--- FAZE 2: Tum model fine-tuning (epoch {FREEZE_EPOCHS + 1}-{EPOCHS}) ---")
unfreeze_backbone(model)
optimizer = optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "classifier" not in n],
     "lr": LR_BACKBONE},
    {"params": model.classifier.parameters(), "lr": LR_CLASSIFIER * 0.1},
], weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=1e-6)

last_completed_epoch = FREEZE_EPOCHS
for epoch in range(FREEZE_EPOCHS + 1, EPOCHS + 1):
    epoch_start = time.time()
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, use_mix=True
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
    scheduler.step()
    train_losses.append(train_loss); train_accs.append(train_acc)
    val_losses.append(val_loss); val_accs.append(val_acc)

    epoch_time = time.time() - epoch_start
    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        marker = " * En iyi!"
        epochs_without_improvement = 0
    else:
        marker = ""
        epochs_without_improvement += 1

    print(
        f"Epoch [{epoch:3d}/{EPOCHS}] | "
        f"Egitim: {train_loss:.4f} / %{train_acc:.2f} | "
        f"Val: {val_loss:.4f} / %{val_acc:.2f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
        f"{epoch_time:.1f}s{marker} | "
        f"iyilesme yok: {epochs_without_improvement}"
    )

    last_completed_epoch = epoch
    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f"\n[!] {EARLY_STOP_PATIENCE} epoch boyunca iyilesme yok. Erken durduruluyor.")
        stopped_early = True
        break

total_time = time.time() - start_time
print("=" * 60)
print(f"Egitim tamamlandi! Toplam sure: {total_time / 60:.1f} dakika")
print(f"En iyi validation dogruluğu: %{best_acc:.2f} (epoch {best_epoch})")
if stopped_early:
    print(f"Erken durduruldu (epoch {last_completed_epoch}/{EPOCHS}).")
print(f"Model kaydedildi: {MODEL_SAVE_PATH}")
print("=" * 60)

# =============================================
# 8. SINIF BAZLI DOGRULUK
# =============================================
print("\n--- Sinif Bazli Dogruluk Analizi ---")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.eval()

labels_path = os.path.join(BASE_DIR, "labels_custom_tr.json")
LABELS_TR = {}
if os.path.exists(labels_path):
    with open(labels_path, "r", encoding="utf-8") as f:
        LABELS_TR = json.load(f)

class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES
with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        for i in range(targets.size(0)):
            label = targets[i].item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

for i in range(NUM_CLASSES):
    acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    tr = LABELS_TR.get(CLASSES[i], CLASSES[i])
    print(f"  {tr:20s} ({CLASSES[i]:15s}): %{acc:.1f} ({class_correct[i]}/{class_total[i]})")

# =============================================
# 9. GRAFIKLER
# =============================================
actual_epochs = len(train_losses)
epochs_range = range(1, actual_epochs + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs_range, train_losses, "b-", label="Egitim Kaybi", alpha=0.8)
axes[0].plot(epochs_range, val_losses, "r-", label="Validation Kaybi", alpha=0.8)
axes[0].axvline(x=FREEZE_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Fine-tune baslangici")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Kayip (Loss)")
axes[0].set_title("V2-Small | Egitim ve Validation Kaybi")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, train_accs, "b-", label="Egitim Dogrulugu", alpha=0.8)
axes[1].plot(epochs_range, val_accs, "r-", label="Validation Dogrulugu", alpha=0.8)
axes[1].axvline(x=FREEZE_EPOCHS, color="gray", linestyle="--", alpha=0.5, label="Fine-tune baslangici")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Dogruluk (%)")
axes[1].set_title(f"V2-Small | En iyi val: %{best_acc:.2f} (epoch {best_epoch})")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(TRAINING_PLOT, dpi=150)
print(f"\nGrafik kaydedildi: {TRAINING_PLOT}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
class_accs = []
class_labels = []
for i in range(NUM_CLASSES):
    acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    class_accs.append(acc)
    class_labels.append(LABELS_TR.get(CLASSES[i], CLASSES[i]))

colors = ['#2ecc71' if acc >= 70 else '#f39c12' if acc >= 50 else '#e74c3c' for acc in class_accs]
bars = ax2.barh(class_labels, class_accs, color=colors)
ax2.set_xlabel("Dogruluk (%)")
ax2.set_title("V2-Small | Sinif Bazli Validation Dogrulugu")
ax2.set_xlim(0, 100)
avg_acc = sum(class_accs) / len(class_accs) if class_accs else 0
ax2.axvline(x=avg_acc, color="blue", linestyle="--", alpha=0.5, label=f"Ortalama: %{avg_acc:.1f}")
ax2.legend(); ax2.invert_yaxis()
for bar, acc in zip(bars, class_accs):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
             f'%{acc:.1f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(CLASS_ACC_PLOT, dpi=150)
print(f"Grafik kaydedildi: {CLASS_ACC_PLOT}")

print("\nTum islemler tamamlandi!")
