"""
ImageNet 1000 etiketini Türkçeye çevir ve kaydet
"""

import json
import urllib.request
from deep_translator import GoogleTranslator

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

print("İngilizce etiketler indiriliyor...")
with urllib.request.urlopen(LABELS_URL) as response:
    labels_en = json.loads(response.read().decode())

print(f"Toplam {len(labels_en)} etiket bulundu.")
print("Türkçeye çevriliyor... (bu birkaç dakika sürebilir)\n")

translator = GoogleTranslator(source='en', target='tr')

# Toplu çeviri (50'şerli gruplar halinde)
labels_tr = {}
batch_size = 50
for i in range(0, len(labels_en), batch_size):
    batch = labels_en[i:i + batch_size]
    combined = "\n".join(batch)
    try:
        translated = translator.translate(combined)
        translated_list = translated.split("\n")
        for j, label in enumerate(batch):
            idx = i + j
            if j < len(translated_list):
                labels_tr[label] = translated_list[j].strip().capitalize()
            else:
                labels_tr[label] = label.capitalize()
    except Exception as e:
        print(f"  Hata (batch {i}): {e}")
        for j, label in enumerate(batch):
            labels_tr[label] = label.capitalize()

    done = min(i + batch_size, len(labels_en))
    print(f"  {done}/{len(labels_en)} çevrildi...")

with open("labels_tr.json", "w", encoding="utf-8") as f:
    json.dump(labels_tr, f, ensure_ascii=False, indent=2)

print(f"\nTamamlandı! {len(labels_tr)} etiket 'labels_tr.json' dosyasına kaydedildi.")
print("\nÖrnekler:")
examples = ["cat", "dog", "car", "airplane", "apple", "banana", "person"]
for ex in examples:
    if ex in labels_tr:
        print(f"  {ex} -> {labels_tr[ex]}")
