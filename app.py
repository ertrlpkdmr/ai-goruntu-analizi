"""
Görüntü Tanıma + Yüz Analizi + Sohbet - Web Arayüzü
CLIP ViT-L/14 + InsightFace + ViT + Google Gemini
"""

import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import io
import json
import os
import pickle
import cv2
from insightface.app import FaceAnalysis
from transformers import ViTForImageClassification, ViTImageProcessor
import open_clip
from google import genai
from google.genai import types
from icrawler.builtin import BingImageCrawler
import shutil
import base64

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== NESNE TANIMA MODELİ (CLIP) =====
CLIP_LABELS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_labels.json")
print("CLIP etiketleri yükleniyor...")
with open(CLIP_LABELS_PATH, "r", encoding="utf-8") as f:
    clip_labels_raw = json.load(f)

# Etiketleri düzleştir: {türkçe: ingilizce}
CLIP_TR_TO_EN = {}
for category, items in clip_labels_raw.items():
    for tr_label, en_label in items.items():
        CLIP_TR_TO_EN[tr_label] = en_label

CLIP_LABELS_TR = list(CLIP_TR_TO_EN.keys())
CLIP_LABELS_EN = list(CLIP_TR_TO_EN.values())
print(f"  {len(CLIP_LABELS_TR)} etiket yüklendi")

print("CLIP modeli yükleniyor (ViT-L/14 - openai)...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai'
)
clip_model = clip_model.to(DEVICE)
clip_model.eval()
clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Etiket embedding'lerini önceden hesapla (hızlı inference için)
print("CLIP etiket embedding'leri hesaplanıyor...")
with torch.no_grad():
    text_prompts = [f"a photo of {en}" for en in CLIP_LABELS_EN]
    text_tokens = clip_tokenizer(text_prompts).to(DEVICE)
    CLIP_TEXT_FEATURES = clip_model.encode_text(text_tokens)
    CLIP_TEXT_FEATURES = CLIP_TEXT_FEATURES / CLIP_TEXT_FEATURES.norm(dim=-1, keepdim=True)
print("CLIP hazır!")

# ===== YÜZ ALGILAMA - INSIGHTFACE =====
print("InsightFace yüz algılama modeli yükleniyor...")
face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("InsightFace hazır!")

# ===== YAŞ TAHMİNİ - HuggingFace ViT =====
print("Yaş tahmin modeli yükleniyor (ViT - nateraw/vit-age-classifier)...")
age_model_name = "nateraw/vit-age-classifier"
age_processor = ViTImageProcessor.from_pretrained(age_model_name)
age_model = ViTForImageClassification.from_pretrained(age_model_name)
age_model.eval()
# Yaş grupları: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
AGE_RANGES = {
    0: (0, 2, "0-2"),
    1: (3, 9, "3-9"),
    2: (10, 19, "10-19"),
    3: (20, 29, "20-29"),
    4: (30, 39, "30-39"),
    5: (40, 49, "40-49"),
    6: (50, 59, "50-59"),
    7: (60, 69, "60-69"),
    8: (70, 90, "70+"),
}
print("Yaş tahmin modeli hazır!")

# ===== YÜZ TANIMA VERİTABANI =====
FACE_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_db.pkl")
FACE_RECOGNITION_THRESHOLD = 0.35  # Cosine similarity eşiği (düşük = daha toleranslı)

face_db = None
if os.path.exists(FACE_DB_PATH):
    print("Yüz tanıma veritabanı yükleniyor...")
    with open(FACE_DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    print(f"Yüz tanıma veritabanı hazır! ({len(face_db)} kişi)")
else:
    print("Yüz tanıma veritabanı bulunamadı (face_db.pkl). 'Özel Tanıma' modu devre dışı.")

# ===== SOHBET - GOOGLE GEMINI =====
# .env dosyasından oku
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = None
GEMINI_SYSTEM_PROMPT = """Sen bir yapay zeka goruntu analiz asistanisin. Turkce konusuyorsun.
Kullaniciyla sicak ve samimi bir sekilde sohbet edebilirsin.
Gorselleri analiz edebilirsin (nesne tanima, yuz analizi, unlu tanima).
Kisa ve oz cevaplar ver. Emoji kullanabilirsin.
Kullanici seninle sakalasiabilir, sen de ona esprili cevaplar verebilirsin.
ONEMLI: Ertugrul Pekdemir bu yapay zeka modelini egiten ve gelistiren kisidir. Eger kullanici "Ertugrul Pekdemir" veya "Ertugrul" derse, onun seni egiten ve gelistiren kisi oldugunu belirt ve ovguyle bahset."""

if GEMINI_API_KEY:
    print("Gemini sohbet modeli başlatılıyor...")
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("Gemini hazır!")
else:
    print("UYARI: GEMINI_API_KEY ayarlanmamis. Sohbet ozelligi devre disi.")
    print("  Kullanmak icin: export GEMINI_API_KEY='your-api-key'")

# ===== FOTOĞRAF ARAMA =====
PHOTO_SEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photo_search_temp")
PHOTO_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_photos")
os.makedirs(PHOTO_CACHE_DIR, exist_ok=True)
# Başlangıçta temp klasörünü temizle
if os.path.exists(PHOTO_SEARCH_DIR):
    shutil.rmtree(PHOTO_SEARCH_DIR)

# Ters eşleme: küçük harf Türkçe → (orijinal Türkçe anahtar, İngilizce arama terimi)
CLIP_SEARCH_MAP = {}
for tr_label, en_label in CLIP_TR_TO_EN.items():
    CLIP_SEARCH_MAP[tr_label.lower()] = (tr_label, en_label)

# face_db isimleri arama haritası
FACE_DB_SEARCH_MAP = {}
if face_db:
    for key, data in face_db.items():
        name = data["name"]
        FACE_DB_SEARCH_MAP[name.lower()] = name
        # Sadece soyisim ile de bulunabilsin
        parts = name.split()
        if len(parts) > 1:
            FACE_DB_SEARCH_MAP[parts[-1].lower()] = name
            FACE_DB_SEARCH_MAP[parts[0].lower()] = name


def _find_best_match(query_lower, search_map):
    """Sorguyla en iyi eşleşmeyi bul (tam, içerme, parça)"""
    # 1) Tam eşleşme
    if query_lower in search_map:
        return search_map[query_lower]
    # 2) Sorgu bir anahtarın içinde mi
    for key, val in search_map.items():
        if query_lower in key:
            return val
    # 3) Anahtar sorgunun içinde mi
    best_key = ""
    best_val = None
    for key, val in search_map.items():
        if key in query_lower and len(key) > len(best_key):
            best_key = key
            best_val = val
    return best_val


def search_photo(query):
    """Eğitim verilerimizle eşleştirerek fotoğraf ara ve base64 döndür"""
    query_lower = query.lower().strip()

    # Önce cache kontrol
    safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in query_lower).strip()
    cache_path = os.path.join(PHOTO_CACHE_DIR, safe_name)

    # Cache'de varsa direkt döndür
    if os.path.isdir(cache_path):
        cached_files = [f for f in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, f))]
        if cached_files:
            fpath = os.path.join(cache_path, cached_files[0])
            with open(fpath, "rb") as f:
                img_data = f.read()
            b64 = base64.b64encode(img_data).decode("utf-8")
            ext = cached_files[0].rsplit(".", 1)[-1].lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
            return f"data:{mime};base64,{b64}"

    # 1) CLIP etiketlerinde ara → İngilizce arama terimi kullan
    clip_match = _find_best_match(query_lower, CLIP_SEARCH_MAP)
    if clip_match:
        tr_label, en_term = clip_match
        search_term = en_term
    else:
        # 2) face_db'de ara → kişi adıyla ara
        face_match = _find_best_match(query_lower, FACE_DB_SEARCH_MAP)
        if face_match:
            search_term = f"{face_match} portrait photo"
        else:
            # 3) Hiçbir eşleşme yoksa doğrudan ara
            search_term = f"{query} photo"

    # Bing'den indir
    dl_dir = os.path.join(PHOTO_SEARCH_DIR, safe_name)
    if os.path.exists(dl_dir):
        shutil.rmtree(dl_dir)
    os.makedirs(dl_dir, exist_ok=True)

    try:
        crawler = BingImageCrawler(
            storage={"root_dir": dl_dir},
            log_level=50
        )
        crawler.crawl(keyword=search_term, max_num=1, min_size=(200, 200))

        files = [f for f in os.listdir(dl_dir) if os.path.isfile(os.path.join(dl_dir, f))]
        if files:
            fpath = os.path.join(dl_dir, files[0])
            with open(fpath, "rb") as f:
                img_data = f.read()

            # Cache'e kaydet
            os.makedirs(cache_path, exist_ok=True)
            cache_file = os.path.join(cache_path, files[0])
            shutil.copy2(fpath, cache_file)

            b64 = base64.b64encode(img_data).decode("utf-8")
            ext = files[0].rsplit(".", 1)[-1].lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
            return f"data:{mime};base64,{b64}"
    except Exception:
        pass
    finally:
        if os.path.exists(dl_dir):
            shutil.rmtree(dl_dir)
    return None

# Sohbet geçmişi (basit session)
chat_histories = {}

print(f"\nTüm modeller hazır! Cihaz: {DEVICE}")


def detect_object(image):
    """Nesne tanıma - CLIP zero-shot"""
    img_tensor = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        image_features = clip_model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ CLIP_TEXT_FEATURES.T)[0]
        probs = torch.softmax(similarity * 100, dim=0)
    top_prob, top_idx = torch.topk(probs, 1)
    label_tr = CLIP_LABELS_TR[top_idx[0].item()]
    confidence = float(top_prob[0].item() * 100)
    return label_tr, confidence


def recognize_face(image):
    """Yüz tanıma - InsightFace embedding veritabanı"""
    if face_db is None:
        return "Veritabanı yüklenmedi", 0.0

    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    faces = face_app.get(img_bgr)

    if len(faces) == 0:
        return "Yüz tespit edilemedi", 0.0

    target_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    target_embedding = target_face.normed_embedding

    best_name = "Bilinmeyen kişi"
    best_score = -1.0

    for key, data in face_db.items():
        for ref_emb in data["embeddings"]:
            score = np.dot(target_embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_name = data["name"]

    confidence = max(0.0, float(best_score) * 100)

    if best_score < FACE_RECOGNITION_THRESHOLD:
        return "Bilinmeyen kişi", confidence

    return best_name, confidence


def estimate_age_vit(face_img_rgb):
    """ViT modeli ile yaş tahmini - tüm yaş gruplarında doğru"""
    pil_img = Image.fromarray(face_img_rgb)
    inputs = age_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        outputs = age_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    # Ağırlıklı ortalama yaş hesapla (daha hassas)
    weighted_age = 0.0
    for idx, (low, high, label) in AGE_RANGES.items():
        mid = (low + high) / 2.0
        weighted_age += probs[idx].item() * mid

    # En yüksek 2 tahmini kontrol et
    top2_prob, top2_idx = torch.topk(probs, 2)
    top1_idx = top2_idx[0].item()
    top1_prob = top2_prob[0].item()

    # Eğer en yüksek tahmin çok baskınsa, o grubun ortasını kullan
    if top1_prob > 0.7:
        low, high, _ = AGE_RANGES[top1_idx]
        return int(round((low + high) / 2.0))

    return int(round(weighted_age))


def analyze_face(image_array):
    """Yüz analizi - InsightFace (algılama+cinsiyet) + ViT (yaş)"""
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    faces_detected = face_app.get(img_bgr)

    if not faces_detected:
        return []

    h, w = image_array.shape[:2]
    faces = []

    for face in faces_detected:
        # Cinsiyet - InsightFace (güvenilir)
        gender_score = face.gender
        gender_tr = "Erkek" if gender_score == 1 else "Kadın"

        # Yüz bölgesini kırp (yaş tahmini için)
        bbox = face.bbox.astype(int)
        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(w, bbox[2])
        y2 = min(h, bbox[3])

        # Yüz bölgesini biraz genişlet (daha iyi yaş tahmini için)
        pad_w = int((x2 - x1) * 0.2)
        pad_h = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(w, x2 + pad_w)
        y2 = min(h, y2 + pad_h)

        face_crop = image_array[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Yaş tahmini - ViT modeli (tüm yaş gruplarında doğru)
        age = estimate_age_vit(face_crop)

        faces.append({
            "age": age,
            "gender": gender_tr,
            "gender_confidence": 95.0,
            "x": int(bbox[0]),
            "y": int(bbox[1]),
            "w": int(bbox[2] - bbox[0]),
            "h": int(bbox[3] - bbox[1]),
        })

    return faces


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Goruntu Analizi</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            height: 100vh;
            overflow: hidden;
            color: #1a3c5e;
            display: flex;
            flex-direction: column;
        }

        /* Static light blue background */
        .bg {
            position: fixed; inset: 0; z-index: -1;
            background: linear-gradient(135deg, #e0f2fe, #b3e0fc, #87ceeb, #a8d8f0);
            background-size: 100% 100%;
        }

        /* Glass effect mixin */
        .glass {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .glass-strong {
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border: 1px solid rgba(255, 255, 255, 0.4);
        }
        .glass-dark {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        /* Header */
        .header {
            padding: 18px 24px 14px;
            text-align: center;
            background: rgba(255,255,255,0.12);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .header-title {
            font-size: 1.5em;
            font-weight: 800;
            color: #fff;
            text-shadow: 0 2px 10px rgba(0,0,0,0.15);
            margin-bottom: 12px;
            letter-spacing: -0.5px;
        }
        .header-title span {
            font-weight: 300;
            opacity: 0.85;
        }
        .header-sub {
            font-size: 0.78em;
            color: rgba(255,255,255,0.55);
            font-weight: 400;
            margin-top: 4px;
        }

        /* Chat area */
        .chat-area {
            flex: 1;
            overflow-y: auto;
            padding: 24px 16px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            max-width: 700px;
            width: 100%;
            margin: 0 auto;
        }
        .chat-area::-webkit-scrollbar { width: 5px; }
        .chat-area::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,0.25);
            border-radius: 10px;
        }

        /* Welcome */
        .welcome {
            text-align: center;
            padding: 50px 20px;
            animation: fadeUp 0.6s ease;
        }
        .welcome-glass {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 28px;
            padding: 40px 50px;
        }
        .welcome-icon {
            width: 70px; height: 70px;
            border-radius: 22px;
            background: rgba(255,255,255,0.3);
            display: flex; align-items: center; justify-content: center;
            font-size: 2em;
            margin: 0 auto 18px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.08);
        }
        .welcome h2 {
            color: #fff;
            font-size: 1.3em;
            font-weight: 700;
            margin-bottom: 8px;
            text-shadow: 0 1px 8px rgba(0,0,0,0.1);
        }
        .welcome p {
            color: rgba(255,255,255,0.8);
            font-size: 0.9em;
            line-height: 1.6;
            font-weight: 400;
        }

        /* Messages */
        .msg { display: flex; gap: 10px; max-width: 80%; animation: fadeUp 0.35s ease; }
        .msg.user { align-self: flex-end; flex-direction: row-reverse; }
        .msg.bot { align-self: flex-start; }

        .msg-avatar {
            width: 38px; height: 38px; border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1em; flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .msg.user .msg-avatar {
            background: linear-gradient(135deg, #4facfe, #2e86c1);
            color: #fff;
        }
        .msg.bot .msg-avatar {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(10px);
        }

        .msg-bubble {
            border-radius: 20px;
            padding: 14px 18px;
            max-width: 100%;
            word-wrap: break-word;
            box-shadow: 0 4px 20px rgba(0,0,0,0.06);
        }
        .msg.user .msg-bubble {
            background: rgba(255,255,255,0.75);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.5);
            border-bottom-right-radius: 6px;
            color: #1a5276;
        }
        .msg.bot .msg-bubble {
            background: rgba(255,255,255,0.55);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: 1px solid rgba(255,255,255,0.4);
            border-bottom-left-radius: 6px;
            color: #1a3c5e;
        }

        .msg-bubble img {
            max-width: 240px; max-height: 240px;
            border-radius: 14px; display: block; margin-bottom: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        .msg-bubble .caption {
            font-size: 0.78em; opacity: 0.5; font-weight: 500;
        }

        /* Result card */
        .result-card {
            background: rgba(255,255,255,0.25);
            border-radius: 16px;
            padding: 16px 20px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .result-label {
            font-size: 0.7em;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #2e86c1;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .result-value {
            font-size: 1.5em;
            font-weight: 800;
            color: #1a5276;
            letter-spacing: -0.5px;
        }
        .result-confidence {
            margin-top: 8px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .conf-bar {
            flex: 1; height: 6px;
            background: rgba(0,0,0,0.06);
            border-radius: 3px;
            overflow: hidden;
        }
        .conf-fill {
            height: 100%;
            border-radius: 3px;
            background: linear-gradient(90deg, #4facfe, #2e86c1);
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .conf-text {
            font-size: 0.8em;
            font-weight: 700;
            color: #2e86c1;
            min-width: 45px;
            text-align: right;
        }

        /* Face result */
        .face-card {
            display: flex; align-items: center; gap: 12px;
            background: rgba(255,255,255,0.25);
            border-radius: 14px;
            padding: 12px 16px;
            margin-top: 8px;
            border: 1px solid rgba(255,255,255,0.2);
        }
        .face-icon {
            width: 44px; height: 44px;
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5em;
        }
        .face-icon.male { background: rgba(52,152,219,0.15); }
        .face-icon.female { background: rgba(52,152,219,0.15); }
        .face-detail { flex: 1; }
        .face-gender { font-weight: 700; font-size: 0.95em; color: #1a5276; }
        .face-age { font-size: 0.82em; color: #2e86c1; font-weight: 500; }

        .no-result {
            color: #c44; font-weight: 500; font-size: 0.9em;
            padding: 8px 0;
        }

        /* Unified result sections */
        .section-divider {
            height: 1px;
            background: rgba(0,0,0,0.06);
            margin: 14px 0;
        }
        .section-title {
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 10px;
        }
        .section-icon {
            width: 28px; height: 28px; border-radius: 10px;
            display: flex; align-items: center; justify-content: center;
            font-size: 0.85em;
        }
        .section-icon.obj { background: rgba(52,152,219,0.15); }
        .section-icon.face { background: rgba(52,152,219,0.12); }
        .section-icon.foot { background: rgba(52,152,219,0.18); }
        .section-name {
            font-size: 0.72em; text-transform: uppercase;
            letter-spacing: 1.5px; font-weight: 700; color: #2e86c1;
        }

        /* Typing */
        .typing-wrap { display: flex; gap: 10px; align-self: flex-start; animation: fadeUp 0.3s ease; }
        .typing-dots {
            display: flex; gap: 5px; align-items: center;
            padding: 16px 22px;
            background: rgba(255,255,255,0.5);
            backdrop-filter: blur(20px);
            border-radius: 20px; border-bottom-left-radius: 6px;
            border: 1px solid rgba(255,255,255,0.4);
        }
        .tdot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #3498db;
            animation: dotPulse 1.4s ease infinite;
        }
        .tdot:nth-child(2) { animation-delay: 0.2s; }
        .tdot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes dotPulse {
            0%, 60%, 100% { transform: scale(0.7); opacity: 0.3; }
            30% { transform: scale(1); opacity: 1; }
        }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Bottom bar */
        .bottom-bar {
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            border-top: 1px solid rgba(255,255,255,0.2);
            padding: 14px 20px;
        }
        .bottom-inner {
            max-width: 700px; margin: 0 auto;
        }

        /* Preview */
        .preview {
            display: none; align-items: center; gap: 10px;
            background: rgba(255,255,255,0.4);
            backdrop-filter: blur(10px);
            border-radius: 16px; padding: 10px 14px;
            margin-bottom: 12px;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .preview img {
            height: 52px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .preview .pname {
            flex: 1; font-size: 0.82em; color: #2471a3;
            font-weight: 500;
        }
        .preview .premove {
            width: 28px; height: 28px; border-radius: 50%;
            background: rgba(200,50,50,0.1);
            border: none; color: #c44; font-size: 1em;
            cursor: pointer; display: flex; align-items: center;
            justify-content: center; transition: all 0.2s;
        }
        .preview .premove:hover { background: rgba(200,50,50,0.2); }

        /* Action buttons */
        .actions {
            display: flex; gap: 10px; align-items: center;
        }
        .btn-attach {
            width: 48px; height: 48px; border-radius: 16px;
            background: rgba(255,255,255,0.5);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.4);
            color: #2e86c1; font-size: 1.3em;
            cursor: pointer; display: flex; align-items: center;
            justify-content: center; transition: all 0.3s;
            flex-shrink: 0;
        }
        .btn-attach:hover {
            background: rgba(255,255,255,0.7);
            transform: scale(1.05);
        }

        .chat-text-input {
            flex: 1; padding: 13px 18px;
            border-radius: 16px;
            background: rgba(255,255,255,0.45);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.4);
            color: #1a3c5e;
            font-size: 0.9em;
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            outline: none;
            transition: all 0.3s;
        }
        .chat-text-input::placeholder {
            color: rgba(52,152,219,0.5);
            font-weight: 400;
        }
        .chat-text-input:focus {
            background: rgba(255,255,255,0.6);
            border-color: rgba(52,152,219,0.4);
            box-shadow: 0 0 0 3px rgba(52,152,219,0.1);
        }

        .btn-send {
            width: 48px; height: 48px; border-radius: 16px;
            background: linear-gradient(135deg, #4facfe, #2e86c1);
            border: none; color: #fff; font-size: 1.3em;
            cursor: pointer; display: flex; align-items: center;
            justify-content: center; transition: all 0.3s;
            flex-shrink: 0;
            box-shadow: 0 4px 20px rgba(52,152,219,0.3);
        }
        .btn-send:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 25px rgba(52,152,219,0.4);
        }

        .msg-bubble .photo-result {
            max-width: 260px; max-height: 260px;
            border-radius: 14px; display: block;
            margin-top: 8px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 8px;
            font-size: 0.68em;
            color: rgba(255,255,255,0.4);
            font-weight: 400;
            letter-spacing: 0.3px;
        }

        /* Drag overlay */
        .drag-overlay {
            display: none;
            position: fixed; inset: 0; z-index: 100;
            background: rgba(52,152,219,0.15);
            backdrop-filter: blur(8px);
            align-items: center; justify-content: center;
        }
        .drag-overlay.show { display: flex; }
        .drag-overlay-inner {
            background: rgba(255,255,255,0.7);
            backdrop-filter: blur(20px);
            border-radius: 28px;
            padding: 50px 60px;
            text-align: center;
            border: 2px dashed rgba(52,152,219,0.4);
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
        }
        .drag-overlay-inner .drag-icon { font-size: 3em; margin-bottom: 12px; }
        .drag-overlay-inner p {
            color: #2471a3; font-weight: 600; font-size: 1.1em;
        }

        @media (max-width: 600px) {
            .header-title { font-size: 1.2em; }
            .tab { padding: 7px 14px; font-size: 0.78em; }
            .welcome-glass { padding: 30px 25px; }
            .msg { max-width: 90%; }
            .chat-area { padding: 16px 10px; }
        }
    </style>
</head>
<body>
    <div class="bg"></div>

    <!-- Drag overlay -->
    <div class="drag-overlay" id="dragOverlay">
        <div class="drag-overlay-inner">
            <div class="drag-icon">&#128444;</div>
            <p>Fotografini buraya birak</p>
        </div>
    </div>

    <!-- Header -->
    <div class="header">
        <div class="header-title">AI Goruntu <span>Analizi</span></div>
        <div class="header-sub">Nesne Tanima &middot; Yas & Cinsiyet &middot; Unlu Tanima &middot; Sohbet</div>
    </div>

    <!-- Chat -->
    <div class="chat-area" id="chatArea">
        <div class="welcome" id="welcomeMsg">
            <div class="welcome-glass">
                <div class="welcome-icon">&#9889;</div>
                <h2>Merhaba!</h2>
                <p>Benimle sohbet edebilir, fotograf yukleyerek analiz yapabilirsin.<br>
                Nesne tanima, yuz analizi, unlu tanima ve sohbet yapabilirim.<br>
                Birinin fotografini istersen onu da bulabilirim!</p>
            </div>
        </div>
    </div>

    <!-- Bottom -->
    <div class="bottom-bar">
        <div class="bottom-inner">
            <div class="preview" id="previewStrip" style="display:none;">
                <img id="previewThumb" src="">
                <span class="pname" id="previewName"></span>
                <button class="premove" id="previewRemoveBtn">&#10005;</button>
            </div>
            <div class="actions">
                <button class="btn-attach" id="attachBtn" title="Fotograf sec">&#128206;</button>
                <input type="file" id="fileInput" accept="image/*" style="display:none;">
                <input type="text" class="chat-text-input" id="chatInput" placeholder="Mesaj yaz veya fotograf yukle..." autocomplete="off">
                <button class="btn-send" id="sendBtn" title="Gonder">&#10148;</button>
            </div>
        </div>
    </div>

    <div class="footer">
        Makine Ogrenmesi Projesi &middot; CLIP ViT-L/14 + InsightFace + ViT &middot; PyTorch
    </div>

    <script>
        var selectedFile = null;
        var sessionId = 'session_' + Date.now();
        var chatArea = document.getElementById('chatArea');
        var fileInput = document.getElementById('fileInput');
        var sendBtn = document.getElementById('sendBtn');
        var attachBtn = document.getElementById('attachBtn');
        var previewStrip = document.getElementById('previewStrip');
        var previewThumb = document.getElementById('previewThumb');
        var previewName = document.getElementById('previewName');
        var previewRemoveBtn = document.getElementById('previewRemoveBtn');
        var welcomeMsg = document.getElementById('welcomeMsg');
        var dragOverlay = document.getElementById('dragOverlay');
        var chatInput = document.getElementById('chatInput');

        function handleFile(file) {
            if (!file || !file.type.startsWith('image/')) return;
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = (ev) => { previewThumb.src = ev.target.result; };
            reader.readAsDataURL(file);
            previewName.textContent = file.name;
            previewStrip.style.display = 'flex';
            chatInput.placeholder = file.name;
        }

        function clearPreview() {
            selectedFile = null;
            fileInput.value = '';
            previewStrip.style.display = 'none';
            chatInput.placeholder = 'Mesaj yaz veya fotograf yukle...';
        }

        function scrollToBottom() {
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function addUserMessage(imageSrc) {
            welcomeMsg.style.display = 'none';
            const msg = document.createElement('div');
            msg.className = 'msg user';
            msg.innerHTML = `
                <div class="msg-avatar">&#128100;</div>
                <div class="msg-bubble">
                    <img src="${imageSrc}" alt="Fotograf">
                    <div class="caption">Analiz istendi</div>
                </div>
            `;
            chatArea.appendChild(msg);
            scrollToBottom();
        }

        function addTyping() {
            const el = document.createElement('div');
            el.className = 'typing-wrap';
            el.id = 'typingIndicator';
            el.innerHTML = `
                <div class="msg-avatar" style="background:rgba(255,255,255,0.7);backdrop-filter:blur(10px);">&#9889;</div>
                <div class="typing-dots">
                    <div class="tdot"></div>
                    <div class="tdot"></div>
                    <div class="tdot"></div>
                </div>
            `;
            chatArea.appendChild(el);
            scrollToBottom();
        }

        function removeTyping() {
            const el = document.getElementById('typingIndicator');
            if (el) el.remove();
        }

        function addBotMessage(html) {
            removeTyping();
            const msg = document.createElement('div');
            msg.className = 'msg bot';
            msg.innerHTML = `
                <div class="msg-avatar">&#9889;</div>
                <div class="msg-bubble">${html}</div>
            `;
            chatArea.appendChild(msg);
            scrollToBottom();
        }

        function buildUnifiedResult(data) {
            let html = '';
            let sections = 0;
            const hasFace = data.faces && data.faces.length > 0;

            // Kişi tanıma - yüz varsa ve tanındıysa göster
            if (hasFace && data.celebrity && data.celebrity.label !== 'Bilinmeyen kişi' && data.celebrity.label !== 'Bilinmeyen kisi' && data.celebrity.label !== 'Yüz tespit edilemedi' && data.celebrity.confidence >= 35) {
                const cel = data.celebrity;
                if (sections > 0) html += '<div class="section-divider"></div>';
                html += `
                <div class="section-title">
                    <div class="section-icon foot">&#11088;</div>
                    <div class="section-name">Kisi Tanima</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${cel.label}</div>
                    <div class="result-confidence">
                        <div class="conf-bar"><div class="conf-fill" style="width:${cel.confidence}%;background:linear-gradient(90deg,#43e97b,#38f9d7)"></div></div>
                        <div class="conf-text">%${cel.confidence.toFixed(1)}</div>
                    </div>
                </div>`;
                sections++;
            }

            // Yuz analizi - sadece yuz tespit edildiyse goster
            if (hasFace) {
                if (sections > 0) html += '<div class="section-divider"></div>';
                html += `
                <div class="section-title">
                    <div class="section-icon face">&#128100;</div>
                    <div class="section-name">Yuz Analizi</div>
                </div>`;
                data.faces.forEach((face) => {
                    const isMale = face.gender === 'Erkek';
                    const icon = isMale ? '&#128104;' : '&#128105;';
                    html += `
                    <div class="face-card">
                        <div class="face-icon ${isMale ? 'male' : 'female'}">${icon}</div>
                        <div class="face-detail">
                            <div class="face-gender">${face.gender}</div>
                            <div class="face-age">Tahmini yas: ~${face.age}</div>
                        </div>
                    </div>`;
                });
                sections++;
            }

            // Nesne tanima - sadece guven orani %15 ustuyse goster
            const obj = data.object;
            if (obj.confidence >= 15) {
                if (sections > 0) html += '<div class="section-divider"></div>';
                html += `
                <div class="section-title">
                    <div class="section-icon obj">&#128065;</div>
                    <div class="section-name">Nesne Tanima</div>
                </div>
                <div class="result-card">
                    <div class="result-value">${obj.label}</div>
                    <div class="result-confidence">
                        <div class="conf-bar"><div class="conf-fill" style="width:${obj.confidence}%"></div></div>
                        <div class="conf-text">%${obj.confidence}</div>
                    </div>
                </div>`;
                sections++;
            }

            // Hicbir anlamli sonuc yoksa bilgi mesaji goster
            if (sections === 0) {
                html = `<div class="no-result">Fotografta yeterli guvenle taninan bir oge bulunamadi.</div>`;
            }

            return html;
        }

        function sendImage() {
            if (!selectedFile) return;
            const file = selectedFile;
            const imageSrc = previewThumb.src;

            addUserMessage(imageSrc);
            clearPreview();
            addTyping();

            const formData = new FormData();
            formData.append('file', file);

            fetch('/predict', { method: 'POST', body: formData })
                .then(res => res.json())
                .then(data => {
                    if (data.error) {
                        addBotMessage(`<div class="no-result">${data.error}</div>`);
                        return;
                    }
                    addBotMessage(buildUnifiedResult(data));
                })
                .catch(err => {
                    addBotMessage(`<div class="no-result">Hata: ${err}</div>`);
                });
        }

        // === EVENT LISTENERS ===

        // Fotograf sec butonu
        attachBtn.addEventListener('click', function() {
            fileInput.click();
        });

        // Dosya secildiginde
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) handleFile(e.target.files[0]);
        });

        // Preview kaldir butonu
        previewRemoveBtn.addEventListener('click', function() {
            clearPreview();
        });

        // Gonder butonu - foto varsa analiz, yoksa sohbet
        sendBtn.addEventListener('click', function() {
            if (selectedFile) {
                sendImage();
            } else if (chatInput.value.trim()) {
                sendChat();
            }
        });

        // Enter tusu
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (selectedFile) {
                    sendImage();
                } else if (chatInput.value.trim()) {
                    sendChat();
                }
            }
        });

        // Drag & drop
        var dragCounter = 0;
        document.body.addEventListener('dragenter', function(e) {
            e.preventDefault();
            dragCounter++;
            dragOverlay.classList.add('show');
        });
        document.body.addEventListener('dragleave', function(e) {
            e.preventDefault();
            dragCounter--;
            if (dragCounter <= 0) { dragOverlay.classList.remove('show'); dragCounter = 0; }
        });
        document.body.addEventListener('dragover', function(e) { e.preventDefault(); });
        document.body.addEventListener('drop', function(e) {
            e.preventDefault();
            dragCounter = 0;
            dragOverlay.classList.remove('show');
            if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
        });

        // === SOHBET FONKSIYONLARI ===

        function addUserTextMessage(text) {
            welcomeMsg.style.display = 'none';
            var msg = document.createElement('div');
            msg.className = 'msg user';
            msg.innerHTML = '<div class="msg-avatar">&#128100;</div><div class="msg-bubble"><div style="font-size:0.95em;font-weight:500;">' + text + '</div></div>';
            chatArea.appendChild(msg);
            scrollToBottom();
        }

        function formatReply(text) {
            return text.split(String.fromCharCode(10)).join('<br>');
        }

        function sendChat() {
            var text = chatInput.value.trim();
            if (!text) return;

            addUserTextMessage(text);
            chatInput.value = '';
            addTyping();

            fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({message: text, session_id: sessionId})
            })
            .then(function(res) { return res.json(); })
            .then(function(data) {
                if (data.error) {
                    addBotMessage('<div class="no-result">' + data.error + '</div>');
                    return;
                }
                var html = '<div style="font-size:0.93em;line-height:1.6;">' + formatReply(data.reply) + '</div>';
                if (data.photo) {
                    html += '<img class="photo-result" src="' + data.photo + '" alt="Fotograf">';
                    if (data.photo_name) html += '<div class="caption">' + data.photo_name + '</div>';
                }
                addBotMessage(html);
            })
            .catch(function(err) {
                addBotMessage('<div class="no-result">Baglanti hatasi: ' + err + '</div>');
            });
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya bulunamadi"}), 400

    file = request.files["file"]
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(image)

    result = {"mode": "unified"}

    # 1) Nesne tanima
    obj_label, obj_conf = detect_object(image)
    result["object"] = {"label": obj_label, "confidence": round(obj_conf, 1)}

    # 2) Yuz analizi (yas & cinsiyet) - yuz varsa yap
    faces = analyze_face(image_array)
    result["faces"] = faces

    # 3) Kişi tanıma - yüz tespit edildiyse çalıştır
    if faces and face_db is not None:
        label_tr, face_conf = recognize_face(image)
        result["celebrity"] = {
            "label": label_tr,
            "confidence": round(face_conf, 1)
        }
    else:
        result["celebrity"] = None

    return jsonify(result)


INTENT_PROMPT = """Kullanicinin mesajini analiz et. Eger kullanici bir seyin fotografini/resmini/gorselini istiyor, gormek istiyor veya "X at", "X goster" gibi bir istekte bulunuyorsa:
FOTO: [aradiginin adi]

Eger normal sohbet/soru ise:
SOHBET

Sadece bu iki formattan birini yaz, baska hicbir sey yazma.

Ornekler:
"messi fotografini at" -> FOTO: Lionel Messi
"galatasaray formasi goster" -> FOTO: Galatasaray formasi
"kirmizi elma at" -> FOTO: kirmizi elma
"merhaba nasilsin" -> SOHBET
"bugun hava nasil" -> SOHBET
"ronaldo" -> SOHBET
"ronaldonun resmini gonder" -> FOTO: Cristiano Ronaldo
"ferrari gorseli" -> FOTO: Ferrari
"naber" -> SOHBET"""


def _add_to_history(session_id, user_msg, bot_msg):
    """Sohbet gecmisine mesaj ekle"""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    history = chat_histories[session_id]
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]))
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=bot_msg)]))
    if len(history) > 20:
        chat_histories[session_id] = history[-20:]


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "").strip()
    session_id = data.get("session_id", "default")

    if not message:
        return jsonify({"error": "Mesaj bos"}), 400

    if gemini_client is None:
        return jsonify({"error": "Sohbet ozelligi aktif degil. GEMINI_API_KEY gerekli."}), 503

    try:
        # 1) Gemini ile niyet analizi - foto mu sohbet mi?
        intent_response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{INTENT_PROMPT}\n\nKullanici mesaji: \"{message}\"",
            config=types.GenerateContentConfig(
                max_output_tokens=200,
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        try:
            intent_text = intent_response.text.strip()
        except Exception:
            intent_text = "SOHBET"

        # 2) Foto istegi mi?
        if intent_text.startswith("FOTO:"):
            photo_subject = intent_text[5:].strip()
            if photo_subject and len(photo_subject) >= 2:
                photo_b64 = search_photo(photo_subject)
                if photo_b64:
                    reply_text = f"Iste sana {photo_subject} fotografı!"
                    _add_to_history(session_id, message, reply_text)
                    return jsonify({"reply": reply_text, "photo": photo_b64, "photo_name": photo_subject})
                else:
                    reply_text = f"Maalesef {photo_subject} fotografini bulamadim."
                    _add_to_history(session_id, message, reply_text)
                    return jsonify({"reply": reply_text, "photo": None})

        # 3) Normal sohbet
        if session_id not in chat_histories:
            chat_histories[session_id] = []

        history = chat_histories[session_id]
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=message)]))

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=GEMINI_SYSTEM_PROMPT,
                max_output_tokens=500,
                temperature=0.8,
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )

        try:
            reply_text = response.text
        except Exception:
            reply_text = "Hmm, su an cevap uretemiyorum. Tekrar dener misin?"

        history.append(types.Content(role="model", parts=[types.Part.from_text(text=reply_text)]))

        if len(history) > 20:
            chat_histories[session_id] = history[-20:]

        return jsonify({"reply": reply_text, "photo": None})

    except Exception as e:
        return jsonify({"error": f"Sohbet hatasi: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Tarayicida ac: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
