# AI Görüntü Analizi

Görüntü tanıma, yüz analizi ve sohbet özelliklerini tek bir web arayüzünde birleştiren bir yapay zekâ uygulaması. Bir fotoğraf yüklendiğinde; içindeki nesneleri tanır, yüzleri tespit edip yaş/cinsiyet/duygu tahmini yapar, sahneyi yorumlar ve istenirse görsel hakkında sohbet eder.

## Özellikler

- **Nesne tanıma** — CLIP (ViT-L/14) ile yüzlerce kategoride sıfır-atış (zero-shot) sınıflandırma
- **Özel model** — EfficientNetV2-L ile eğitilmiş 20 sınıflık tanıma (futbolcular, takımlar, spor dalları)
- **Yüz analizi** — InsightFace ile yüz tespiti + ViT ile yaş/cinsiyet/duygu tahmini
- **Ünlü/kişi tanıma** — yüz veritabanı (face_db) ile eşleştirme
- **Model Yorumu** — yerel modellerin çıktısından kural-tabanlı, internetsiz çalışan görsel yorumu
- **Gemini Yorumu** — (opsiyonel) Google Gemini ile zengin görsel yorumlama ve sohbet

## Gereksinimler

- **Python 3.10+** (3.12 ile test edildi)
- İnternet bağlantısı (ilk çalıştırmada modeller indirilir)
- **İsteğe bağlı:** NVIDIA GPU + CUDA (yoksa CPU'da da çalışır, sadece daha yavaş)

## Kurulum

Önce depoyu klonla:

```bash
git clone https://github.com/ertrlpkdmr/ai-goruntu-analizi.git
cd ai-goruntu-analizi
```

### Windows (PowerShell)

> Python kurulu değilse: `winget install Python.Python.3.12` ya da
> https://www.python.org/downloads/ (kurulumda **"Add Python to PATH"** işaretle).

```powershell
py -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> `Activate.ps1` "execution policy" hatası verirse bir kez şunu çalıştır, sonra tekrar dene:
> `Set-ExecutionPolicy -Scope Process -Bypass`

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Gemini API anahtarı (opsiyonel ama önerilir)

Sohbet ve "Gemini Yorumu" özellikleri için bir Google Gemini API anahtarı gerekir. Anahtar olmadan da uygulama çalışır; sadece bu iki özellik devre dışı kalır (nesne tanıma, yüz analizi ve "Model Yorumu" yine çalışır).

Proje kök dizininde `.env` adında bir dosya oluştur:

```
GEMINI_API_KEY=buraya_anahtarini_yaz
```

> Anahtarı ücretsiz almak için: https://aistudio.google.com/apikey

## Model dosyaları hakkında (önemli)

Bazı dosyalar boyutları nedeniyle (GitHub 100 MB sınırı) depoya **dâhil edilmemiştir**:

| Dosya | Ne işe yarar | Yoksa ne olur |
|-------|--------------|----------------|
| `best_custom_model.pth` | 20 sınıflık özel model | Özel tanıma devre dışı kalır, diğer her şey çalışır |
| `face_db.pkl` | Yüz tanıma veritabanı | "Özel kişi tanıma" devre dışı kalır |

**Uygulama bu dosyalar olmadan da sorunsuz açılır** ve temel özellikler (CLIP nesne tanıma, yüz/yaş/duygu analizi, Gemini yorumu) çalışır. Bu dosyalar gerekiyorsa proje sahibinden ayrıca temin edin ve proje kök dizinine koyun. (`face_db.pkl`'i `build_face_db.py`, özel modeli `train_custom.py` ile yeniden de üretebilirsiniz.)

> CLIP, InsightFace ve yaş tahmin modeli **otomatik olarak internetten indirilir** (ilk çalıştırmada birkaç GB; sonraki çalıştırmalarda önbellekten gelir).

## Çalıştırma

```bash
python app.py
```

Modeller yüklendikten sonra (ilk seferde birkaç dakika sürebilir) tarayıcıda aç:

```
http://localhost:5000
```

Bir fotoğraf sürükleyip bırak veya yükle; analiz sonuçları ve yorumlar otomatik gelir.

## Proje yapısı

```
app.py                 # Ana web uygulaması (Flask)
clip_labels.json       # CLIP nesne etiketleri (TR/EN)
custom_classes.json    # Özel modelin 20 sınıfı
celebrity_labels.json  # Ünlü/kişi etiketleri
train_custom.py        # Özel modeli eğitme
build_face_db.py       # Yüz veritabanı oluşturma
predict*.py            # Komut satırından tahmin scriptleri
evaluate_model.py      # Model değerlendirme
requirements.txt       # Python bağımlılıkları
```

## Teknolojiler

PyTorch · CLIP (open_clip) · InsightFace · Hugging Face Transformers (ViT) · Google Gemini · Flask
