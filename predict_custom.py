"""
Kendi eğittiğimiz ÖZEL modeli (EfficientNet V2) tek görselle test et.
Takım/sporcu/spor dalı tanıma — 10 sınıf.

Kullanım:
  python predict_custom.py resim.jpg
  python predict_custom.py resim.jpg --arch v2s --weights best_custom_model_v2s.pth
"""

import sys
import os
import json
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Özel eğitilmiş model ile görsel tahmini")
parser.add_argument("images", nargs="+", help="Tahmin edilecek görsel dosyaları")
parser.add_argument("--arch", choices=["v2l", "v2s"], default="v2l",
                    help="Model mimarisi: v2l (Large) veya v2s (Small)")
parser.add_argument("--weights", default="best_custom_model.pth",
                    help="Eğitilmiş ağırlık dosyası (.pth)")
args = parser.parse_args()

# --- Sınıf isimleri (eğitimde kaydedildi) ---
with open(os.path.join(BASE_DIR, "custom_classes.json"), encoding="utf-8") as f:
    CLASSES = json.load(f)

# Türkçe görünen isimler (varsa)
tr_path = os.path.join(BASE_DIR, "labels_custom_tr.json")
LABELS_TR = {}
if os.path.exists(tr_path):
    with open(tr_path, encoding="utf-8") as f:
        LABELS_TR = json.load(f)

NUM_CLASSES = len(CLASSES)

# --- Modeli kur (eğitimdekiyle birebir aynı yapı) ---
if args.arch == "v2l":
    model = models.efficientnet_v2_l(weights=None)
else:
    model = models.efficientnet_v2_s(weights=None)

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, NUM_CLASSES)
)

weights_path = os.path.join(BASE_DIR, args.weights)
state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model = model.to(DEVICE).eval()

print(f"Model yüklendi: {args.weights} ({args.arch}) | cihaz: {DEVICE} | {NUM_CLASSES} sınıf")

# --- Görsel ön işleme (eğitimdeki val_transform ile aynı) ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Tahmin ---
for path in args.images:
    if not os.path.exists(path):
        print(f"\n[!] Dosya bulunamadı: {path}")
        continue

    image = Image.open(path).convert("RGB")
    x = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, min(3, NUM_CLASSES))

    print(f"\n{'='*45}")
    print(f"Görsel: {path}")
    print(f"{'='*45}")
    for i in range(top_probs.size(0)):
        cls = CLASSES[top_idx[i].item()]
        name = LABELS_TR.get(cls, cls)
        conf = top_probs[i].item() * 100
        bar = "█" * int(conf / 5)
        marker = "  <-- TAHMİN" if i == 0 else ""
        print(f"  {name:15s} %{conf:5.1f}  {bar}{marker}")
