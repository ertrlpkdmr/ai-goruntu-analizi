"""
CIFAR-100 Görüntü Sınıflandırma - Transfer Learning ile ResNet50
Makine Öğrenmesi Dersi Projesi
Mixup, CutMix, Label Smoothing ve gelişmiş augmentation ile eğitim
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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
BATCH_SIZE = 64  # Daha küçük batch = daha iyi genelleme
EPOCHS = 100  # 100 sınıf için daha uzun eğitim
LEARNING_RATE = 0.05  # SGD için daha yüksek LR
WARMUP_EPOCHS = 5  # İlk 5 epoch warmup
NUM_WORKERS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
MODEL_SAVE_PATH = "./best_model.pth"
LABELS_CIFAR100_TR = "./labels_cifar100_tr.json"

# Mixup / CutMix ayarları
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 0.5  # %50 Mixup, %50 CutMix

print(f"Kullanılan cihaz: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================
# 2. VERİ HAZIRLAMA (Geliştirilmiş Data Augmentation)
# =============================================

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),  # Cutout benzeri - parça silme
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
])

print("\nCIFAR-100 veri seti indiriliyor (100 sınıf)...")
train_dataset = datasets.CIFAR100(
    root=DATA_DIR, train=True, download=True, transform=train_transform
)
test_dataset = datasets.CIFAR100(
    root=DATA_DIR, train=False, download=True, transform=test_transform
)
CLASSES = train_dataset.classes

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

print(f"Eğitim verisi: {len(train_dataset)} görüntü")
print(f"Test verisi: {len(test_dataset)} görüntü")
print(f"Sınıf sayısı: {len(CLASSES)}")

# =============================================
# 3. MIXUP VE CUTMIX FONKSİYONLARI
# =============================================

def mixup_data(x, y, alpha=0.2):
    """Mixup: İki resmi karıştırarak yeni eğitim verisi oluşturur"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """CutMix: Bir resmin bir parçasını başka bir resimle değiştirir"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

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

    # Lambda'yı gerçek kesim oranına göre güncelle
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup/CutMix için kayıp hesaplama"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================
# 4. MODEL OLUŞTURMA (Transfer Learning - ResNet50)
# =============================================

print("\nModel yükleniyor (ResNet50 - Transfer Learning)...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# CIFAR-100 için son katmanı değiştir
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 100)
)

# CIFAR-100 resimleri 32x32 olduğu için ilk conv katmanını uyarlıyoruz
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Toplam parametre: {total_params:,}")
print(f"Eğitilebilir parametre: {trainable_params:,}")

# =============================================
# 5. KAYIP FONKSİYONU VE OPTİMİZER
# =============================================

# Label Smoothing: Modelin aşırı güvenli yanlış tahminlerini azaltır
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# SGD + Momentum: Görüntü sınıflandırmada Adam'dan genelde daha iyi
optimizer = optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True
)

# Warmup + Cosine Annealing LR
warmup_scheduler = LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
)
cosine_scheduler = CosineAnnealingLR(
    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-5
)
scheduler = SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[WARMUP_EPOCHS]
)

# =============================================
# 6. EĞİTİM VE TEST FONKSİYONLARI
# =============================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixup veya CutMix uygula (ilk 10 epoch'tan sonra)
        use_mix = epoch > 10
        if use_mix:
            if np.random.random() < MIXUP_PROB:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, MIXUP_ALPHA)
            else:
                inputs, targets_a, targets_b, lam = cutmix_data(inputs, targets, CUTMIX_ALPHA)

        optimizer.zero_grad()
        outputs = model(inputs)

        if use_mix:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)

        loss.backward()
        # Gradient clipping - eğitim stabilitesi için
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if use_mix:
            correct += (lam * predicted.eq(targets_a).sum().float()
                       + (1 - lam) * predicted.eq(targets_b).sum().float()).item()
        else:
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
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

    accuracy = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

# =============================================
# 7. EĞİTİM DÖNGÜSÜ
# =============================================

print("\n" + "=" * 60)
print("EĞİTİM BAŞLIYOR")
print(f"Model: ResNet50 | Epoch: {EPOCHS} | Batch: {BATCH_SIZE}")
print(f"LR: {LEARNING_RATE} | Warmup: {WARMUP_EPOCHS} epoch")
print(f"Mixup: α={MIXUP_ALPHA} | CutMix: α={CUTMIX_ALPHA}")
print(f"Label Smoothing: 0.1")
print("=" * 60)

train_losses = []
train_accs = []
test_losses = []
test_accs = []
best_acc = 0.0

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, DEVICE, epoch
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    epoch_time = time.time() - epoch_start

    # En iyi modeli kaydet
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        marker = " ★ En iyi!"
    else:
        marker = ""

    print(
        f"Epoch [{epoch:3d}/{EPOCHS}] | "
        f"Eğitim: {train_loss:.4f} / %{train_acc:.2f} | "
        f"Test: {test_loss:.4f} / %{test_acc:.2f} | "
        f"LR: {current_lr:.6f} | "
        f"{epoch_time:.1f}s{marker}"
    )

total_time = time.time() - start_time
print("=" * 60)
print(f"Eğitim tamamlandı! Toplam süre: {total_time / 60:.1f} dakika")
print(f"En iyi test doğruluğu: %{best_acc:.2f}")
print("=" * 60)

# =============================================
# 8. SINIF BAZLI DOĞRULUK ANALİZİ
# =============================================

print("\n--- Sınıf Bazlı Doğruluk Analizi ---")
model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
model.eval()

NUM_CLASSES = 100
class_correct = [0] * NUM_CLASSES
class_total = [0] * NUM_CLASSES

# Confusion matrix için
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

        for i in range(targets.size(0)):
            label = targets[i].item()
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Türkçe etiketleri yükle
labels_tr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels_cifar100_tr.json")
if os.path.exists(labels_tr_path):
    with open(labels_tr_path, "r", encoding="utf-8") as f:
        LABELS_TR = json.load(f)
else:
    LABELS_TR = {}

# Tüm sınıfları doğruluğa göre sırala
class_accs = []
for i in range(NUM_CLASSES):
    acc = 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    class_accs.append(acc)

sorted_idx = np.argsort(class_accs)

# En kötü 20 sınıf (en çok sorun yaşananlar)
print("\n🔴 En düşük doğruluklu 20 sınıf:")
for i in sorted_idx[:20]:
    tr = LABELS_TR.get(CLASSES[i], CLASSES[i])
    acc = class_accs[i]
    print(f"  {tr:20s} ({CLASSES[i]:18s}): %{acc:.1f} ({class_correct[i]}/{class_total[i]})")

# En iyi 10 sınıf
print("\n🟢 En yüksek doğruluklu 10 sınıf:")
for i in sorted_idx[-10:][::-1]:
    tr = LABELS_TR.get(CLASSES[i], CLASSES[i])
    acc = class_accs[i]
    print(f"  {tr:20s} ({CLASSES[i]:18s}): %{acc:.1f} ({class_correct[i]}/{class_total[i]})")

# Önemli sınıflar - meyveler
print("\n🍎 Meyve ve sebze sınıfları:")
important = ["apple", "orange", "pear", "sweet_pepper", "mushroom"]
for name in important:
    if name in CLASSES:
        i = CLASSES.index(name)
        tr = LABELS_TR.get(name, name)
        acc = class_accs[i]
        print(f"  {tr:20s} ({name:18s}): %{acc:.1f} ({class_correct[i]}/{class_total[i]})")

# Ortalama doğruluk
avg_acc = sum(class_accs) / len(class_accs)
print(f"\nOrtalama sınıf doğruluğu: %{avg_acc:.2f}")

# En çok karışan sınıf çiftlerini bul
print("\n🔄 En çok karışan sınıf çiftleri:")
all_preds = np.array(all_preds)
all_targets = np.array(all_targets)

confusion_pairs = {}
for true_label, pred_label in zip(all_targets, all_preds):
    if true_label != pred_label:
        key = (CLASSES[true_label], CLASSES[pred_label])
        confusion_pairs[key] = confusion_pairs.get(key, 0) + 1

sorted_confusions = sorted(confusion_pairs.items(), key=lambda x: -x[1])[:15]
for (true_cls, pred_cls), count in sorted_confusions:
    true_tr = LABELS_TR.get(true_cls, true_cls)
    pred_tr = LABELS_TR.get(pred_cls, pred_cls)
    print(f"  {true_tr} → {pred_tr} ({true_cls} → {pred_cls}): {count} hata")

# =============================================
# 9. GRAFİKLER
# =============================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, EPOCHS + 1), train_losses, "b-", label="Eğitim Kaybı", alpha=0.8)
axes[0].plot(range(1, EPOCHS + 1), test_losses, "r-", label="Test Kaybı", alpha=0.8)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Kayıp (Loss)")
axes[0].set_title("Eğitim ve Test Kaybı")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, EPOCHS + 1), train_accs, "b-", label="Eğitim Doğruluğu", alpha=0.8)
axes[1].plot(range(1, EPOCHS + 1), test_accs, "r-", label="Test Doğruluğu", alpha=0.8)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Doğruluk (%)")
axes[1].set_title("Eğitim ve Test Doğruluğu")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
print("\nGrafik 'training_results.png' olarak kaydedildi.")

# Sınıf bazlı doğruluk grafiği - en kötü 15 + en iyi 10
worst_idx = sorted_idx[:15]
best_idx = sorted_idx[-10:][::-1]
show_idx = list(best_idx) + list(worst_idx)

fig2, ax2 = plt.subplots(figsize=(14, 8))
labels_show = [LABELS_TR.get(CLASSES[i], CLASSES[i]) for i in show_idx]
accs_show = [class_accs[i] for i in show_idx]

colors = ['#2ecc71' if acc >= 70 else '#f39c12' if acc >= 50 else '#e74c3c' for acc in accs_show]
bars = ax2.barh(labels_show, accs_show, color=colors)
ax2.set_xlabel("Doğruluk (%)")
ax2.set_title("Sınıf Bazlı Test Doğruluğu (En İyi 10 + En Kötü 15)")
ax2.set_xlim(0, 100)
ax2.axvline(x=avg_acc, color="blue", linestyle="--", alpha=0.5, label=f"Ortalama: %{avg_acc:.1f}")
ax2.legend()
ax2.invert_yaxis()
for bar, acc in zip(bars, accs_show):
    ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'%{acc:.1f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig("class_accuracy.png", dpi=150)
print("Grafik 'class_accuracy.png' olarak kaydedildi.")

print("\nTüm işlemler tamamlandı!")
