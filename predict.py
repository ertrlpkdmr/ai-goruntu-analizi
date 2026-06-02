"""
Kendi resminle modeli test et!
Kullanım: python predict.py resim.jpg
CIFAR-100: 100 sınıf - ResNet50 ile Transfer Learning
"""

import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json

# Ayarlar
MODEL_PATH = "./best_model.pth"
LABELS_TR_PATH = "./labels_cifar100_tr.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 sınıf isimleri (İngilizce - alfabetik sıra)
CLASSES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark",
    "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel",
    "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone",
    "television", "tiger", "tractor", "train", "trout", "tulip", "turtle",
    "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]

# Türkçe etiketler
if os.path.exists(LABELS_TR_PATH):
    with open(LABELS_TR_PATH, "r", encoding="utf-8") as f:
        LABELS_TR = json.load(f)
else:
    LABELS_TR = {c: c for c in CLASSES}

# CIFAR-100 normalizasyonu - Test Time Augmentation (TTA) desteği
base_transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
])

# TTA: Orijinal + yatay flip ile ortalama tahmin
tta_flip_transform = transforms.Compose([
    transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    ),
])


def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 100)
    )
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def predict(image_path, model):
    image = Image.open(image_path).convert("RGB")

    # Test Time Augmentation: orijinal + flip ortalaması
    input_orig = base_transform(image).unsqueeze(0).to(DEVICE)
    input_flip = tta_flip_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out_orig = torch.softmax(model(input_orig), dim=1)
        out_flip = torch.softmax(model(input_flip), dim=1)
        probabilities = (out_orig + out_flip) / 2.0
        probabilities = probabilities[0]

    # En yüksek 5 tahmini göster
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    print(f"\nResim: {image_path}")
    print("-" * 50)
    for i in range(5):
        cls_en = CLASSES[top5_idx[i].item()]
        cls_tr = LABELS_TR.get(cls_en, cls_en)
        prob = top5_prob[i].item() * 100
        bar = "█" * int(prob / 2)
        print(f"  {i+1}. {cls_tr:18s} : %{prob:5.1f} {bar}")
    print("-" * 50)
    best_tr = LABELS_TR.get(CLASSES[top5_idx[0].item()], CLASSES[top5_idx[0].item()])
    confidence = top5_prob[0].item() * 100
    print(f"  Sonuç: Bu bir {best_tr}! (Güven: %{confidence:.1f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py resim1.jpg [resim2.png ...]")
        print("\nÖrnek:")
        print("  python predict.py kedi.jpg")
        print("  python predict.py araba.png ucak.jpg")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"Hata: Model dosyası bulunamadı ({MODEL_PATH})")
        print("Önce train.py ile modeli eğitin.")
        sys.exit(1)

    model = load_model()
    print("Model yüklendi! (ResNet50)")

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"\nHata: '{path}' dosyası bulunamadı!")
            continue
        predict(path, model)
