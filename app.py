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
# VRAM tasarrufu için fp16 (4 GB'lık GPU'larda bellek dostu)
if DEVICE.type == "cuda":
    clip_model = clip_model.half()
clip_model.eval()
# CLIP'in aktüel device/dtype'ı
CLIP_DEVICE = next(clip_model.parameters()).device
CLIP_DTYPE = next(clip_model.parameters()).dtype
clip_tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Etiket embedding'lerini önceden hesapla (hızlı inference için)
print("CLIP etiket embedding'leri hesaplanıyor...")
with torch.no_grad():
    text_prompts = [f"a photo of {en}" for en in CLIP_LABELS_EN]
    text_tokens = clip_tokenizer(text_prompts).to(CLIP_DEVICE)
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

# ===== ÖZEL EĞITILMIŞ NESNE/SPORCU MODELI (EfficientNet V2 Large) =====
CUSTOM_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_custom_model.pth")
# Sınıf sırası eğitimdeki (alfabetik) sırayla aynı olmalı — custom_classes.json'dan yüklenir
CUSTOM_CLASSES = [
    "barcelona", "basketbol", "bayernmunich", "bellingham", "besiktas",
    "fenerbahce", "futbol", "galatasaray", "haaland", "liverpool",
    "manchester_city", "mbappe", "messi", "neymar", "realmadrid",
    "ronaldo", "tenis", "trabzonspor", "vinicius", "voleybol",
]
# Sınıf adlarından kullanıcıya gösterilecek Türkçe etikete harita
CUSTOM_LABELS_DISPLAY = {
    "barcelona": "FC Barcelona",
    "basketbol": "basketbol",
    "bayernmunich": "Bayern Münih",
    "bellingham": "Jude Bellingham",
    "besiktas": "Beşiktaş JK",
    "fenerbahce": "Fenerbahçe SK",
    "futbol": "futbol",
    "galatasaray": "Galatasaray SK",
    "haaland": "Erling Haaland",
    "liverpool": "Liverpool FC",
    "manchester_city": "Manchester City",
    "mbappe": "Kylian Mbappé",
    "messi": "Lionel Messi",
    "neymar": "Neymar Jr",
    "realmadrid": "Real Madrid",
    "ronaldo": "Cristiano Ronaldo",
    "tenis": "tenis",
    "trabzonspor": "Trabzonspor",
    "vinicius": "Vinícius Júnior",
    "voleybol": "voleybol",
}
custom_model = None
CUSTOM_TRANSFORM = None

if os.path.exists(CUSTOM_MODEL_PATH):
    print("Özel nesne/sporcu modeli yükleniyor (EfficientNet V2 Large)...")
    from torchvision import models as _tvmodels, transforms as _tvtransforms
    import torch.nn as _tnn
    _custom = _tvmodels.efficientnet_v2_l(weights=None)
    _in = _custom.classifier[1].in_features
    _custom.classifier = _tnn.Sequential(
        _tnn.Dropout(0.3),
        _tnn.Linear(_in, len(CUSTOM_CLASSES)),
    )
    _state = torch.load(CUSTOM_MODEL_PATH, map_location=DEVICE, weights_only=True)
    _custom.load_state_dict(_state)
    _custom = _custom.to(DEVICE)
    _custom.eval()
    custom_model = _custom
    CUSTOM_TRANSFORM = _tvtransforms.Compose([
        _tvtransforms.Resize(256),
        _tvtransforms.CenterCrop(224),
        _tvtransforms.ToTensor(),
        _tvtransforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print(f"Özel model hazır! ({len(CUSTOM_CLASSES)} sınıf: {', '.join(CUSTOM_LABELS_DISPLAY.values())})")
else:
    print("best_custom_model.pth bulunamadı — özel tanıma devre dışı.")


# ===== Threading kilidi =====
import threading
# GPU'yu tüm CLIP çağrıları arasında serileştirir (4GB VRAM'de çakışmayı önler)
# RLock çünkü helper fonksiyonlar birbirini çağırabilir (analyze_image_extra -> clip_classify)
GPU_LOCK = threading.RLock()


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
GEMINI_SYSTEM_PROMPT = """Sen uzman bir yapay zeka goruntu analiz asistanisin. Turkce konusuyorsun.
Kullaniciyla sicak, samimi ve bilgilendirici bir sekilde iletisim kurarsın.

GORSEL YORUMLAMA KURALLARI:
- Gorselde gorduklerini zengin ve detayli acikla
- Nesnelerin ne ise yaradigini, ne anlama geldigini mutlaka belirt
- Manzara/doga: mevsimi, havayi, isigi, atmosferi ve duyguyu yorumla
- Yemek/icecek: adi, kulturel baglami, icerigi, nasil hazirlandi
- Hayvan: turu, ozellikleri, davranisi, yasam alani
- Arac/teknoloji: marka, model, ne ise yaradigi, ozellikler
- Kisi/portre: ifadeyi, duyguyu, ortami, kiyafetten cikarimlari yorumla
- Mimari/bina: tarz, donem, islevsellik
- Sanat/tasarim: teknik, stil, mesaj
- Dogal ve akici Turkce yaz, sanki bir arkadas anlatiyormus gibi
- Teknik jargondan kacin, herkesin anlayacagi bir dil kullan
- Markdown bold (**baslik**) kullanarak yapilandir

SOHBET KURALLARI:
- Kisa ve oz cevaplar ver, gereksiz uzatma
- Emoji kullanabilirsin ama asiri kullanma
- Espri yapabilirsin, samimi ol
- Kullanici gorsel hakkinda takip sorusu sorarsa, onceki gorseli referans al
- Kullanicinin dil seviyesine uyum sagla

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

import random


def clip_classify(image, labels_en, labels_tr, prompt_template="a photo of {}"):
    """CLIP ile genel sınıflandırma yardımcısı"""
    with GPU_LOCK, torch.no_grad():
        prompts = [prompt_template.format(en) for en in labels_en]
        tokens = clip_tokenizer(prompts).to(CLIP_DEVICE)
        text_feats = clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        img_tensor = clip_preprocess(image).unsqueeze(0).to(CLIP_DEVICE, dtype=CLIP_DTYPE)
        img_feats = clip_model.encode_image(img_tensor)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        sims = (img_feats @ text_feats.T)[0]
        probs = torch.softmax(sims * 100, dim=0)
    results = []
    top_probs, top_idxs = torch.topk(probs, min(3, len(labels_en)))
    for i in range(len(top_idxs)):
        results.append({"label": labels_tr[top_idxs[i].item()], "confidence": float(top_probs[i].item() * 100)})
    return results


def analyze_image_extra(image):
    """CLIP ile renk, aktivite, stil, mevsim, ışık, ortam analizi"""
    extra = {}

    # Renk paleti
    color_en = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray", "golden", "colorful", "pastel", "dark", "bright"]
    color_tr = ["kırmızı", "mavi", "yeşil", "sarı", "turuncu", "mor", "pembe", "kahverengi", "siyah", "beyaz", "gri", "altın sarısı", "rengarenk", "pastel", "koyu tonlar", "parlak renkler"]
    color_res = clip_classify(image, color_en, color_tr, "a {} colored photo")
    extra["colors"] = [r for r in color_res if r["confidence"] >= 25]

    # Aktivite/eylem
    activity_en = ["eating", "drinking", "walking", "running", "sitting", "standing", "dancing", "playing sports",
                   "reading", "working on a laptop", "cooking", "sleeping", "talking", "laughing", "driving", "swimming",
                   "taking a selfie", "shopping", "traveling", "studying", "painting", "playing music", "exercising at the gym",
                   "crying", "hugging", "kissing", "talking on the phone", "playing video games", "riding a bicycle",
                   "hiking outdoors", "camping", "fishing", "gardening", "cleaning", "giving a presentation",
                   "teaching", "celebrating", "praying", "climbing", "skateboarding", "shopping at a market", "posing for a photo"]
    activity_tr = ["yemek yeme", "içecek içme", "yürüme", "koşma", "oturma", "ayakta durma", "dans etme", "spor yapma",
                   "okuma", "bilgisayarda çalışma", "yemek pişirme", "uyuma", "konuşma", "gülme", "araç kullanma", "yüzme",
                   "selfie çekme", "alışveriş yapma", "seyahat etme", "ders çalışma", "resim yapma", "müzik çalma", "spor salonunda egzersiz",
                   "ağlama", "sarılma", "öpüşme", "telefonla konuşma", "video oyunu oynama", "bisiklete binme",
                   "doğa yürüyüşü", "kamp yapma", "balık tutma", "bahçeyle uğraşma", "temizlik yapma", "sunum yapma",
                   "ders anlatma", "kutlama yapma", "ibadet etme", "tırmanma", "kaykay yapma", "pazarda alışveriş", "poz verme"]
    act_res = clip_classify(image, activity_en, activity_tr, "a person {}")
    extra["activities"] = [r for r in act_res if r["confidence"] >= 35]

    # Fotoğraf stili
    style_en = ["professional photography", "amateur snapshot", "artistic photo", "documentary photo",
                "portrait photography", "street photography", "food photography", "product photography",
                "wildlife photography", "macro photography", "black and white photo", "HDR photo", "vintage photo", "minimalist photo",
                "aerial drone photography", "night photography", "long exposure photo", "bokeh blurred background photo",
                "panorama wide photo", "fashion photography", "architecture photography", "sports action photography", "astrophotography night sky"]
    style_tr = ["profesyonel fotoğraf", "günlük çekim", "sanatsal fotoğraf", "belgesel tarzı",
                "portre çekimi", "sokak fotoğrafçılığı", "yemek fotoğrafı", "ürün fotoğrafı",
                "doğa/yaban hayatı", "makro çekim", "siyah beyaz", "HDR çekim", "retro/vintage", "minimalist",
                "havadan drone çekimi", "gece çekimi", "uzun pozlama", "bokeh arka plan",
                "panorama geniş açı", "moda fotoğrafı", "mimari fotoğraf", "spor aksiyon çekimi", "astrofotoğrafi"]
    style_res = clip_classify(image, style_en, style_tr, "a {}")
    extra["style"] = style_res[0] if style_res and style_res[0]["confidence"] >= 30 else None

    # Mevsim
    season_en = ["spring", "summer", "autumn", "winter"]
    season_tr = ["ilkbahar", "yaz", "sonbahar", "kış"]
    season_res = clip_classify(image, season_en, season_tr, "a photo taken in {}")
    extra["season"] = season_res[0] if season_res and season_res[0]["confidence"] >= 40 else None

    # Işık durumu
    light_en = ["natural light", "artificial light", "backlight", "soft light", "harsh light", "dim light", "neon light", "candlelight",
                "golden hour light", "blue hour light", "bright sunlight", "overcast cloudy light", "studio lighting", "moonlight"]
    light_tr = ["doğal ışık", "yapay ışık", "arka ışık", "yumuşak ışık", "sert ışık", "loş ortam", "neon ışık", "mum ışığı",
                "altın saat ışığı", "mavi saat ışığı", "parlak güneş ışığı", "bulutlu ışık", "stüdyo ışığı", "ay ışığı"]
    light_res = clip_classify(image, light_en, light_tr, "a photo with {}")
    extra["light"] = light_res[0] if light_res and light_res[0]["confidence"] >= 35 else None

    # Ortam/mekan detayı
    place_en = ["living room", "bedroom", "kitchen", "bathroom", "office", "classroom", "restaurant", "cafe",
                "park", "beach", "garden", "street", "highway", "bridge", "stadium", "hospital", "museum",
                "airport", "train station", "shopping mall", "gym", "church", "mosque", "forest trail", "rooftop",
                "library", "supermarket", "open market bazaar", "factory", "parking lot", "swimming pool area",
                "ski resort", "mountain top", "lakeside", "desert", "cave", "subway station", "inside an airplane",
                "concert venue", "wedding hall", "hair salon", "bakery shop", "butcher shop", "terrace balcony",
                "construction site", "farm field", "harbor port", "amusement park"]
    place_tr = ["oturma odası", "yatak odası", "mutfak", "banyo", "ofis", "sınıf", "restoran", "kafe",
                "park", "plaj", "bahçe", "sokak", "otoyol", "köprü", "stadyum", "hastane", "müze",
                "havalimanı", "tren istasyonu", "alışveriş merkezi", "spor salonu", "kilise", "cami", "orman yolu", "çatı katı",
                "kütüphane", "süpermarket", "açık pazar", "fabrika", "otopark", "havuz başı",
                "kayak merkezi", "dağ zirvesi", "göl kenarı", "çöl", "mağara", "metro istasyonu", "uçak içi",
                "konser alanı", "düğün salonu", "kuaför", "fırın", "kasap", "teras balkon",
                "inşaat alanı", "tarla", "liman", "lunapark"]
    place_res = clip_classify(image, place_en, place_tr, "a photo of a {}")
    extra["place"] = place_res[0] if place_res and place_res[0]["confidence"] >= 30 else None

    return extra


def generate_offline_interpretation(obj_results, faces, emotions, scene_tags, celebrity=None, user_text=None, extra=None, object_uncertain=False):
    """Yerel modellerimizin sonuçlarıyla Türkçe görsel yorumu üret - Gemini gerekmez.

    object_uncertain=True ise (CLIP ham kosinüsü düşük), nesne adı GÜVENİLMEZ
    demektir; bu durumda nesneyi kesin bir dille adlandırmak yerine emin
    olmadığımızı söyleriz -- listede olmayan bir nesneyi yanlış isimle (örn.
    mousepad'i 'şarj cihazı') iddialı anlatmamak için.
    """
    if extra is None:
        extra = {}
    parts = []

    # --- TUTARLILIK FİLTRELERİ ---
    # İnsan gerektiren sahneleri yüz yoksa at (selfie, grup fotografi)
    _people_scenes = {"selfie", "grup fotografi"}
    if not faces:
        scene_tags = [s for s in (scene_tags or []) if s not in _people_scenes]

    # Ana nesne yemek kategorisinde mi?
    _food_keywords = {"elma", "armut", "portakal", "muz", "çilek", "karpuz", "üzüm",
                      "domates", "biber", "mantar", "pizza", "hamburger", "makarna",
                      "ekmek", "pasta", "kek", "kurabiye", "dondurma", "çorba",
                      "salata", "yemek", "meyve", "sebze", "tatlı", "peynir"}
    main_obj_label = (obj_results[0]["label"].lower() if obj_results else "")
    is_food_object = any(kw in main_obj_label for kw in _food_keywords)

    # Dış mekan sahnesi var mı? (mevsim ancak dış mekanda mantıklı)
    _outdoor_scenes = {"dis mekan", "doga", "sehir", "deniz", "dag", "orman",
                       "gece", "gun batimi", "gun dogumu", "manzara", "havadan cekim",
                       "karli", "yagmurlu", "gunesli", "gunduz"}
    has_outdoor = any(s in _outdoor_scenes for s in (scene_tags or []))
    # Dış mekan yoksa mevsim bilgisini iptal et (mantıksız olmasın)
    if not has_outdoor and extra.get("season"):
        extra = dict(extra)
        extra.pop("season", None)

    # --- 1) Sahne Tanımı - zengin ve doğal ---
    scene_openers = {
        "ic mekan": ["Bir iç mekanda çekilmiş bu görselde", "Kapalı bir alanda çekilmiş bu karede"],
        "dis mekan": ["Açık havada çekilmiş bu görselde", "Dış mekanda çekilmiş bu karede"],
        "doga": ["Doğanın kucağında çekilmiş bu görselde", "Yeşillikler arasında çekilmiş bu karede"],
        "sehir": ["Şehir manzarasının eşlik ettiği bu görselde", "Kent dokusunun hissedildiği bu karede"],
        "deniz": ["Denizin masmavi sularının görüldüğü bu görselde", "Deniz kenarında çekilmiş bu karede"],
        "dag": ["Dağların ihtişamının görüldüğü bu görselde", "Yüksek zirvelerin eşlik ettiği bu karede"],
        "orman": ["Ağaçların arasından süzülen bu görselde", "Ormanın derinliklerinden gelen bu karede"],
        "gece": ["Gecenin karanlığında çekilmiş bu görselde", "Gece vakti yakalanan bu karede"],
        "gunduz": ["Gün ışığında çekilmiş bu görselde", "Aydınlık bir ortamda yakalanan bu karede"],
        "gun batimi": ["Gün batımının büyüleyici ışığında çekilmiş bu görselde", "Ufukta batan güneşin renklendirdiği bu karede"],
        "gun dogumu": ["Şafağın ilk ışıklarıyla çekilmiş bu görselde", "Gün doğumunun taze enerjisiyle dolu bu karede"],
        "yagmurlu": ["Yağmurun eşlik ettiği bu görselde", "Islak ve yağmurlu bir ortamda çekilmiş bu karede"],
        "karli": ["Karla kaplı bir ortamda çekilmiş bu görselde", "Beyaz örtünün sardığı bu karede"],
        "gunesli": ["Güneşin aydınlattığı bu görselde", "Güneşli ve berrak bir havada çekilmiş bu karede"],
        "selfie": ["Selfie olarak çekilmiş bu görselde", "Kendi kendine çekilmiş bu karede"],
        "grup fotografi": ["Bir grup halinde çekilmiş bu görselde", "Toplu olarak poz verilen bu karede"],
        "manzara": ["Geniş bir manzaranın görüldüğü bu görselde", "Nefes kesen bir manzarayı yakalayan bu karede"],
        "yakin cekim": ["Yakın çekim olarak yakalanan bu görselde", "Detayların öne çıktığı bu karede"],
        "havadan cekim": ["Kuşbakışı çekilmiş bu görselde", "Havadan yakalanan bu karede"],
        "stüdyo cekimi": ["Stüdyo ortamında profesyonelce çekilmiş bu görselde", "Kontrollü ışık altında çekilmiş bu karede"],
    }
    opener = ""
    if scene_tags:
        for tag in scene_tags:
            if tag in scene_openers:
                opener = random.choice(scene_openers[tag])
                break
    if not opener:
        opener = "Bu görselde"

    # Mekan detayı ekle
    place_str = ""
    if extra.get("place"):
        place_str = f" Mekan olarak **{extra['place']['label']}** görünümü hakim."

    # Mevsim + ışık
    env_details = []
    if extra.get("season"):
        season_desc = {
            "ilkbahar": "İlkbaharın tazeliği hissediliyor",
            "yaz": "Yazın sıcaklığı ve canlılığı ortama yansımış",
            "sonbahar": "Sonbaharın hüzünlü ama güzel renkleri görülüyor",
            "kış": "Kışın soğuk ve dingin atmosferi hakim",
        }
        s = extra["season"]["label"]
        if s in season_desc:
            env_details.append(season_desc[s])
    if extra.get("light"):
        light_desc = {
            "doğal ışık": "doğal ışık ortamı güzelce aydınlatıyor",
            "yapay ışık": "yapay aydınlatma kullanılmış",
            "arka ışık": "arka ışık ilginç bir siluet etkisi yaratıyor",
            "yumuşak ışık": "yumuşak bir ışık hoş bir atmosfer oluşturuyor",
            "sert ışık": "güçlü ışık keskin gölgeler oluşturmuş",
            "loş ortam": "loş bir aydınlatma gizemli bir hava katıyor",
            "neon ışık": "neon ışıklar modern bir görünüm veriyor",
            "mum ışığı": "mum ışığının sıcak tonu romantik bir hava katıyor",
        }
        l = extra["light"]["label"]
        if l in light_desc:
            env_details.append(light_desc[l])

    env_str = ""
    if env_details:
        env_str = " " + ", ".join(env_details) + "."

    # --- 2) Nesne yorumu - çok daha zengin ---
    nesne_text = ""
    if object_uncertain:
        # CLIP ham kosinüsü düşük: nesne listemizde net bir karşılığı yok.
        # Yanlış isim uydurmak yerine dürüst davran.
        nesne_text = ("Görseldeki ana nesneyi tam olarak çıkaramadım; "
                      "elimizdeki tanıma listesinde buna net bir karşılık bulamadım")
    elif obj_results:
        top = obj_results[0]
        if top["confidence"] >= 15:
            # Geniş nesne açıklama veritabanı
            nesne_db = {
                # Özel model sınıfları — sporcu, takım, spor dalı
                "lionel messi": ["futbol tarihinin en büyük isimlerinden **Lionel Messi**", "Arjantinli efsanevi futbolcu **Lionel Messi**"],
                "cristiano ronaldo": ["Portekizli dünya yıldızı **Cristiano Ronaldo**", "modern futbolun ikonik ismi **Cristiano Ronaldo**"],
                "galatasaray sk": ["Türk futbolunun köklü kulüplerinden **Galatasaray SK**", "sarı-kırmızı renkleriyle **Galatasaray**"],
                "fenerbahçe sk": ["Türk futbolunun büyük kulüplerinden **Fenerbahçe SK**", "sarı-lacivert renkleriyle **Fenerbahçe**"],
                "beşiktaş jk": ["siyah-beyaz renkleriyle **Beşiktaş JK**", "Türk futbolunun büyük kulüplerinden **Beşiktaş**"],
                "trabzonspor": ["Karadeniz'in temsilcisi **Trabzonspor**", "bordo-mavi renkleriyle **Trabzonspor**"],
                "futbol": ["bir **futbol** sahnesi", "**futbol** oynandığı bir kare"],
                "tenis": ["bir **tenis** sahnesi", "**tenis** oynandığı bir kare"],
                "voleybol": ["bir **voleybol** sahnesi", "**voleybol** oynandığı bir kare"],
                "kedi": ["tüyleri ve zarif duruşuyla dikkat çeken sevimli bir kedi", "evcil bir kedi, insanların en sevilen dostlarından biri"],
                "köpek": ["sadık bakışlarıyla bir köpek", "insanın en vefalı dostu olan bir köpek"],
                "araba": ["bir otomobil, günlük ulaşımın vazgeçilmezi", "bir araç görülüyor"],
                "bisiklet": ["çevre dostu bir ulaşım aracı olan bisiklet", "hem spor hem ulaşım aracı olarak kullanılan bir bisiklet"],
                "uçak": ["gökyüzünün devasa yolcusu bir uçak", "havacılık mühendisliğinin bir eseri olan bir uçak"],
                "gemi": ["denizlerin üzerinde süzülen bir gemi", "denizcilik dünyasından etkileyici bir gemi"],
                "tren": ["raylar üzerinde ilerleyen bir tren", "toplu taşımanın önemli bir parçası olan bir tren"],
                "otobüs": ["şehir içi ulaşımın bel kemiği bir otobüs", "toplu taşıma aracı olan bir otobüs"],
                "kuş": ["kanatlarını açmış özgür bir kuş", "doğanın en güzel canlılarından bir kuş"],
                "at": ["güçlü ve asil yapısıyla bir at", "insanlık tarihinin en önemli yoldaşlarından biri olan bir at"],
                "çiçek": ["renkleriyle göz kamaştıran bir çiçek", "doğanın en güzel hediyelerinden bir çiçek"],
                "ağaç": ["dallarını gökyüzüne uzatan bir ağaç", "doğanın akciğerleri olan bir ağaç"],
                "pizza": ["İtalyan mutfağının dünyaca ünlü lezzeti pizza", "üzerindeki malzemelerle iştah açıcı bir pizza"],
                "hamburger": ["katmanlarıyla göz dolduran bir hamburger", "fast food kültürünün simgesi bir hamburger"],
                "kahve": ["aromasıyla insanı cezbeden bir fincan kahve", "günün enerjisini veren bir kahve"],
                "futbol topu": ["dünyanın en popüler sporunun simgesi bir futbol topu", "milyonlarca insanı birleştiren futbolun simgesi"],
                "basketbol": ["turuncu rengiyle dikkat çeken bir basketbol topu", "hızlı ve heyecanlı bir sporun aracı"],
                "bilgisayar": ["dijital çağın vazgeçilmezi bir bilgisayar", "iş ve eğlencenin merkezi olan bir bilgisayar"],
                "telefon": ["modern yaşamın olmazsa olmazı bir telefon", "iletişimin en temel aracı olan bir telefon"],
                "kitap": ["sayfalarında bilgi ve hikayeler barındıran bir kitap", "bilgiye açılan bir kapı olan kitap"],
                "saat": ["zamanın akışını gösteren bir saat", "hem işlevsel hem estetik bir aksesuar olan saat"],
                "gözlük": ["görüşü netleştiren veya stili tamamlayan bir gözlük", "hem sağlık hem moda aksesuarı olan gözlük"],
                "çanta": ["günlük eşyaları taşımak için kullanılan bir çanta", "pratik ve şık bir çanta"],
                "televizyon": ["görsel dünyanın penceresi olan bir televizyon", "eğlence ve bilgi kaynağı bir televizyon"],
                "gitar": ["telleriyle melodiler çıkaran bir gitar", "müziğin en romantik enstrümanlarından gitar"],
                "piyano": ["tuşlarıyla büyüleyen bir piyano", "klasik ve modern müziğin temel enstrümanı piyano"],
                "deniz": ["sonsuz mavisiyle büyüleyen deniz", "ufukta uzanan engin deniz manzarası"],
                "göl": ["sakin suların yansıma yaptığı bir göl", "huzur veren durgun bir göl manzarası"],
                "yemek": ["iştah açıcı görünümüyle bir yemek", "özenle hazırlanmış lezzetli bir yemek"],
                "masa": ["üzerinde çeşitli eşyaların bulunduğu bir masa", "günlük hayatın merkezi olan bir masa"],
                "sandalye": ["oturmak için tasarlanmış bir sandalye", "dinlenme ve çalışma alanının parçası bir sandalye"],
                "lamba": ["ortamı aydınlatan bir lamba", "sıcak ışığıyla ortama ambiyans katan bir lamba"],
                "bulut": ["gökyüzünde süzülen pamuk gibi bulutlar", "gökyüzünü süsleyen bulut formasyonları"],
                "kar": ["her yeri beyaza bürüyen kar", "beyaz örtüsüyle büyüleyici bir kar manzarası"],
                "güneş": ["ışığıyla her şeyi aydınlatan güneş", "gökyüzünün en parlak yıldızı güneş"],
                "ay": ["gecenin aydınlatıcısı ay", "karanlıkta parlayan gizemli ay"],
                "çimen": ["yemyeşil bir çimen alanı", "taze kesilmiş gibi görünen yeşil çimenler"],
            }
            label_lower = top["label"].lower()
            if label_lower in nesne_db:
                nesne_text = random.choice(nesne_db[label_lower])
            else:
                nesne_text = f"**{top['label']}** olarak tanımlanan bir öğe"

            # Diğer olasılıklar — sadece ana nesne özel model kaynağından gelMİYORSA göster
            # (Özel model custom model alternatifleri ve yanlış CLIP çıktılarını yönlendirir)
            _custom_labels_lower = {v.lower() for v in CUSTOM_LABELS_DISPLAY.values()}
            if label_lower in _custom_labels_lower:
                others = []  # özel model emin → sadece top-1 göster
            else:
                others = [r for r in obj_results[1:4] if r["confidence"] >= 10]
            if others:
                other_names = [r["label"] for r in others]
                if len(other_names) == 1:
                    nesne_text += f". Arka planda {other_names[0]} de seçiliyor"
                else:
                    nesne_text += f". Ayrıca {', '.join(other_names[:-1])} ve {other_names[-1]} de fark ediliyor"

    # --- 3) Kişi yorumu ---
    kisi_text = ""
    if faces:
        face_count = len(faces)
        if face_count == 1:
            f = faces[0]
            yas_grubu = ""
            age = f["age"]
            if age <= 5: yas_grubu = "küçük bir çocuk"
            elif age <= 12: yas_grubu = "bir çocuk"
            elif age <= 18: yas_grubu = "genç bir " + ("kız" if f["gender"] == "Kadın" else "erkek")
            elif age <= 30: yas_grubu = "genç bir " + ("kadın" if f["gender"] == "Kadın" else "erkek")
            elif age <= 50: yas_grubu = "orta yaşlı bir " + ("kadın" if f["gender"] == "Kadın" else "erkek")
            elif age <= 65: yas_grubu = "olgun yaşta bir " + ("kadın" if f["gender"] == "Kadın" else "erkek")
            else: yas_grubu = "yaşlı bir " + ("kadın" if f["gender"] == "Kadın" else "erkek")

            kisi_text = f"yaklaşık {age} yaşında {yas_grubu} yer alıyor"

            # Duygu
            if emotions and len(emotions) > 0:
                emo = emotions[0]
                duygu_zengin = {
                    "mutlu": [". Yüzündeki gülümseme mutluluğunu açıkça yansıtıyor", ". Neşeli bir ifadeyle gülümsüyor"],
                    "uzgun": [". Bakışlarında hafif bir hüzün seziliyor", ". İfadesinde derin bir düşünce veya üzüntü var"],
                    "kizgin": [". İfadesi ciddi ve biraz kızgın görünüyor", ". Kararlı ve sert bir bakışı var"],
                    "saskin": [". Şaşkın bir ifadeyle bakıyor", ". Gözlerindeki şaşkınlık dikkat çekiyor"],
                    "korkmus": [". İfadesinde endişe veya tedirginlik okunuyor", ". Biraz gergin bir hali var"],
                    "igrenmis": [". Yüzünde hoşnutsuz bir ifade var", ". İfadesi memnuniyetsizlik yansıtıyor"],
                    "notr": [". Sakin ve dengeli bir ifadeye sahip", ". Yüz ifadesi nötr ve dingin görünüyor"],
                    "ozguvenli": [". Duruşu ve bakışları özgüven dolu", ". Kararlı ve kendinden emin bir ifade taşıyor"],
                    "yorgun": [". İfadesinde hafif bir yorgunluk seziliyor", ". Biraz yorgun ama dingin bir hali var"],
                }
                if emo["emotion"] in duygu_zengin:
                    kisi_text += random.choice(duygu_zengin[emo["emotion"]])
                else:
                    kisi_text += f". İfadesi {emo['emotion']} görünüyor"
        else:
            kisi_text = f"{face_count} kişi bir arada görülüyor"
            yas_listesi = []
            for i, f in enumerate(faces):
                gender = "kadın" if f["gender"] == "Kadın" else "erkek"
                emo_str = ""
                if i < len(emotions):
                    emo_str = f", {emotions[i]['emotion']} ifadeli"
                yas_listesi.append(f"~{f['age']} yaş {gender}{emo_str}")
            kisi_text += " (" + ", ".join(yas_listesi) + ")"

        # Ünlü tanıma
        if celebrity and celebrity.get("label") not in (None, "Bilinmeyen kişi", "Bilinmeyen kisi", "Yüz tespit edilemedi") and celebrity.get("confidence", 0) >= 35:
            kisi_text += f". Bu kişi **{celebrity['label']}** olarak tanındı"

    # --- 4) Aktivite --- (eşik yüksek + tutarlılık: yüz yoksa insan aktivitesi yok)
    aktivite_text = ""
    _human_activities = {"yemek yeme", "içecek içme", "yürüme", "koşma", "oturma",
                         "ayakta durma", "dans etme", "spor yapma", "okuma",
                         "çalışma", "yemek pişirme", "uyuma", "konuşma", "gülme",
                         "araç kullanma", "yüzme", "fotoğraf çekme", "alışveriş yapma",
                         "ders çalışma", "resim yapma", "müzik çalma", "egzersiz yapma"}
    if extra.get("activities") and extra["activities"][0]["confidence"] >= 55:
        act = extra["activities"][0]["label"]
        # Yemek nesnesi için "yemek yeme" aktivitesi tekrarcı — zaten bilgi veriyor
        if is_food_object and act in ("yemek yeme", "yemek pişirme"):
            act = None
        # İnsan aktivitesi ama yüz yok → mantıksız, atla
        elif act in _human_activities and not faces:
            act = None
        aktivite_map = {
            "yemek yeme": "Yemek yendiği anlaşılıyor",
            "içecek içme": "Bir şeyler içiliyor",
            "yürüme": "Yürüyüş yapılıyor",
            "koşma": "Koşu yapılıyor, sportif bir an",
            "oturma": "Rahat bir şekilde oturuluyor",
            "ayakta durma": "Ayakta durularak poz veriliyor",
            "dans etme": "Dans edildiği bir an yakalanmış",
            "spor yapma": "Sportif bir aktivite gerçekleştiriliyor",
            "okuma": "Okuma yapılıyor, bilgiyle dolu bir an",
            "çalışma": "Çalışma anı yakalanmış",
            "yemek pişirme": "Mutfakta yemek hazırlanıyor",
            "uyuma": "Dinlenme veya uyku anı",
            "konuşma": "Sohbet edildiği bir an yakalanmış",
            "gülme": "Neşeli bir kahkaha anı yakalanmış",
            "araç kullanma": "Araç kullanıldığı görülüyor",
            "yüzme": "Suda yüzme aktivitesi yapılıyor",
            "fotoğraf çekme": "Fotoğraf çekme anı",
            "alışveriş yapma": "Alışveriş yapıldığı görülüyor",
            "seyahat etme": "Seyahat halinde bir an yakalanmış",
            "ders çalışma": "Ders çalışılıyor, akademik bir an",
            "resim yapma": "Sanatsal bir çalışma, resim yapılıyor",
            "müzik çalma": "Müzik çalındığı bir an yakalanmış",
            "egzersiz yapma": "Egzersiz yapılıyor, sağlıklı bir yaşam",
        }
        if act in aktivite_map:
            aktivite_text = aktivite_map[act]

    # --- 5) Stil ---
    stil_text = ""
    if extra.get("style") and extra["style"]["confidence"] >= 15:
        stil = extra["style"]["label"]
        stil_map = {
            "profesyonel fotoğraf": "Profesyonel bir dokunuşla çekilmiş bu kare oldukça etkileyici.",
            "günlük çekim": "Günlük yaşamdan doğal bir kare yakalanmış.",
            "sanatsal fotoğraf": "Sanatsal bir bakış açısıyla çekilmiş dikkat çekici bir kare.",
            "belgesel tarzı": "Belgesel tarzında, gerçekliği yansıtan bir çekim.",
            "portre çekimi": "Portre tarzında, kişinin öne çıktığı bir çekim.",
            "sokak fotoğrafçılığı": "Sokak fotoğrafçılığı tarzında, yaşamın akışından bir an.",
            "yemek fotoğrafı": "İştah açıcı bir yemek fotoğrafı.",
            "ürün fotoğrafı": "Ürünü ön plana çıkaran profesyonel bir çekim.",
            "doğa/yaban hayatı": "Doğanın güzelliğini yakalayan etkileyici bir çekim.",
            "makro çekim": "Detayların büyütüldüğü etkileyici bir makro çekim.",
            "siyah beyaz": "Siyah beyaz tonlarıyla nostaljik bir atmosfer yaratılmış.",
            "HDR çekim": "HDR tekniğiyle renk ve detaylar zenginleştirilmiş.",
            "retro/vintage": "Retro bir havası olan, nostalji kokan bir kare.",
            "minimalist": "Sade ve minimalist bir kompozisyonla çekilmiş.",
        }
        if stil in stil_map:
            stil_text = stil_map[stil]

    # --- 6) Atmosfer ve ruh hali ---
    atmosfer_text = ""
    atmosfer_candidates = []
    if scene_tags:
        atmosfer_map = {
            "gun batimi": ["Gün batımının sıcak tonları görsele romantik ve huzurlu bir atmosfer katıyor.", "Ufuktaki renk cümbüşü büyüleyici bir tablo oluşturmuş."],
            "gun dogumu": ["Gün doğumunun taze ışığı yeni başlangıçların enerjisini taşıyor.", "Şafağın ilk ışıkları umut dolu bir atmosfer yaratıyor."],
            "gece": ["Gecenin karanlığı gizemli ve büyüleyici bir atmosfer oluşturmuş.", "Gece vakti çekilmiş bu kare, karanlığın kendine has güzelliğini yansıtıyor."],
            "karli": ["Karın beyazlığı sakin ve büyüleyici bir kış atmosferi yaratmış.", "Beyaz örtü her yeri sarmış, masalsı bir görüntü oluşturmuş."],
            "yagmurlu": ["Yağmurun ıslaklığı görsele melankolik ama güzel bir hava katıyor.", "Yağmur damlaları kendine has bir romantizm yaratmış."],
            "doga": ["Doğanın dingin güzelliği görselden açıkça hissediliyor.", "Yeşilin her tonu bir arada, huzur veren bir doğa manzarası."],
            "manzara": ["Gözün gördüğü her yer farklı bir güzellik sunuyor.", "Geniş perspektif nefes kesen bir manzara ortaya koymuş."],
            "sehir": ["Şehrin canlı ve dinamik enerjisi görsele yansımış.", "Kent yaşamının ritmi bu karede hissediliyor."],
            "deniz": ["Denizin sonsuz mavisi ve huzuru görsele hakim.", "Dalgaların ve ufuk çizgisinin birleştiği büyüleyici bir deniz manzarası."],
            "gunesli": ["Güneşin ışıltısı her şeyi aydınlatarak canlı ve enerjik bir ortam yaratmış.", "Parlak güneş ışığı görsele pozitif bir enerji katıyor."],
        }
        for tag in scene_tags:
            if tag in atmosfer_map:
                atmosfer_candidates.extend(atmosfer_map[tag])
    # Renk bazlı atmosfer
    if extra.get("colors"):
        dominant = extra["colors"][0]["label"]
        renk_atmosfer = {
            "altın sarısı": "Altın sarısı tonlar görsele lüks ve sıcak bir hava katıyor.",
            "pastel": "Pastel renkler yumuşak ve sakin bir atmosfer oluşturmuş.",
            "koyu tonlar": "Koyu tonlar görsele derinlik ve ciddiyet katıyor.",
            "parlak renkler": "Parlak ve canlı renkler enerjik bir atmosfer yaratmış.",
            "rengarenk": "Renklerin çeşitliliği canlı ve neşeli bir ortam oluşturmuş.",
        }
        if dominant in renk_atmosfer:
            atmosfer_candidates.append(renk_atmosfer[dominant])

    if atmosfer_candidates:
        # Tekrarları ele, en fazla 2 farklı cümle birleştir → daha dolu bir atmosfer yorumu
        _seen_atmos = []
        for _c in atmosfer_candidates:
            if _c not in _seen_atmos:
                _seen_atmos.append(_c)
        random.shuffle(_seen_atmos)
        atmosfer_text = " ".join(_seen_atmos[:2])

    # === CÜMLE BİRLEŞTİRME - Doğal paragraf oluştur ===
    paragraphs = []

    # Paragraf 1: Sahne + Mekan + Çevre
    p1 = opener
    if nesne_text:
        # Ek bilgiler nokta ile eklenmiş olabilir → "görülüyor"u ana nesneden sonra koy
        if "." in nesne_text:
            main_part, _, extra_part = nesne_text.partition(".")
            p1 += f" {main_part.strip()} görülüyor. {extra_part.strip()}."
        else:
            p1 += f" {nesne_text} görülüyor."
    elif kisi_text:
        p1 += f" {kisi_text}."
    else:
        p1 += " ilgi çekici bir sahne yakalanmış."
    if place_str:
        p1 += place_str
    if env_str:
        p1 += env_str
    paragraphs.append(f"**Sahne Tanımı:** {p1}")

    # Paragraf 2: Detay analizi (nesne + kişi ayrı ayrı)
    detail_parts = []
    if nesne_text and kisi_text:
        detail_parts.append(f"Görseldeki ana öğe olarak {nesne_text}.")
        detail_parts.append(f"Ayrıca {kisi_text}.")
    elif kisi_text and not nesne_text:
        pass  # zaten p1'de var
    if aktivite_text:
        detail_parts.append(aktivite_text + ".")
    if detail_parts:
        paragraphs.append(f"**Detaylar:** {' '.join(detail_parts)}")

    # Paragraf 3: Atmosfer
    if atmosfer_text:
        paragraphs.append(f"**Atmosfer:** {atmosfer_text}")

    # Paragraf 4: Fotoğraf stili
    if stil_text:
        paragraphs.append(f"**Çekim Stili:** {stil_text}")

    # Paragraf 5: Renk paleti
    if extra.get("colors") and len(extra["colors"]) >= 2:
        color_names = [c["label"] for c in extra["colors"][:3]]
        renk_his = {
            "altın sarısı": " Sıcak ve davetkâr bir his uyandırıyor.",
            "pastel": " Yumuşak ve dingin bir denge sağlıyor.",
            "koyu tonlar": " Görsele ciddiyet ve derinlik katıyor.",
            "parlak renkler": " Enerjik ve dikkat çekici bir izlenim bırakıyor.",
            "rengarenk": " Canlı ve neşeli bir hava katıyor.",
            "mavi": " Serinlik ve huzur hissi veriyor.",
            "yeşil": " Doğallık ve tazelik çağrıştırıyor.",
            "kırmızı": " Tutku ve hareket enerjisi taşıyor.",
        }
        dom = extra["colors"][0]["label"]
        renk_p = f"**Renk Paleti:** Görselde baskın olarak {', '.join(color_names)} tonları öne çıkıyor."
        renk_p += renk_his.get(dom, "")
        paragraphs.append(renk_p)

    # Paragraf 6: Genel izlenim / çıkarım — sinyalleri birleştirip yorum yap
    izlenim = []
    _sport_kw = {"futbol", "tenis", "voleybol", "basketbol", "messi", "ronaldo",
                 "galatasaray", "fenerbahçe", "beşiktaş", "trabzonspor", "barcelona",
                 "real madrid", "bayern", "liverpool", "manchester", "neymar",
                 "mbappé", "mbappe", "haaland", "bellingham", "vinícius", "vinicius"}
    if any(kw in main_obj_label for kw in _sport_kw):
        izlenim.append("Kare, spor tutkusunun ve rekabetin enerjisini yansıtıyor; "
                       "bu tür görüntüler çoğu zaman heyecan, aidiyet ve takım ruhu duygusuyla ilişkilendirilir.")
    elif is_food_object:
        izlenim.append("Yemek temalı bu kare, lezzet ve paylaşım kültürünü çağrıştırıyor; "
                       "sofranın insanları bir araya getiren sıcak atmosferini hissettiriyor.")
    elif faces and emotions:
        emo0 = emotions[0]["emotion"]
        izlenim.append(f"İnsan unsurunun öne çıktığı bu karede **{emo0}** ifade, "
                       "fotoğrafın duygusal tonunu belirleyen en güçlü detay olarak göze çarpıyor.")
    elif has_outdoor:
        izlenim.append("Doğal ortamın hakim olduğu bu kare; dinginlik, açık hava ferahlığı ve "
                       "an'ı yaşama duygusunu öne çıkarıyor.")
    # Çekildiği bağlama dair ek çıkarım
    if extra.get("style") and extra["style"].get("label") == "profesyonel fotoğraf":
        izlenim.append("Kompozisyon ve ışık kullanımı, karenin özenle ve bilinçli bir gözle çekildiğini düşündürüyor.")
    if izlenim:
        paragraphs.append(f"**Genel İzlenim:** {' '.join(izlenim)}")

    if len(paragraphs) == 0:
        return "Görselde ilginç bir sahne var ancak detaylı bir yorum oluşturmak için yeterli bilgi edinemedim."

    return "\n\n".join(paragraphs)


def detect_object(image, top_k=5):
    """Nesne tanıma - CLIP zero-shot, top-K sonuç"""
    with GPU_LOCK, torch.no_grad():
        img_tensor = clip_preprocess(image).unsqueeze(0).to(CLIP_DEVICE, dtype=CLIP_DTYPE)
        image_features = clip_model.encode_image(img_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ CLIP_TEXT_FEATURES.T)[0]
        probs = torch.softmax(similarity * 100, dim=0)
    top_probs, top_idxs = torch.topk(probs, top_k)
    results = []
    for i in range(top_k):
        idx = top_idxs[i].item()
        results.append({
            "label": CLIP_LABELS_TR[idx],
            "label_en": CLIP_LABELS_EN[idx],
            "confidence": float(top_probs[i].item() * 100),
            # Ham kosinüs benzerligi: softmax'tan bagimsiz, "gercekten benziyor mu?"
            # sinyali. softmax 961 sinif uzerinde sismis guven uretir (yanlis
            # eslesmede bile %60+), ama ham kosinus dogru (~0.24+) ile zayif/yanlis
            # (~0.20) eslesmeyi ayirir. Belirsizlik karari bunun uzerinden verilir.
            "cosine": float(similarity[idx].item()),
        })
    return results


def detect_custom_object(image, top_k=3):
    """Özel eğitilmiş model (EfficientNet V2 L) ile sporcu/takım/spor dalı tanı.
    Model sadece 10 sınıfta eğitilmiş — her zaman bir sonuç döner, gerçek karar için
    çağıranın güven skorunu kontrol etmesi gerekir."""
    if custom_model is None or CUSTOM_TRANSFORM is None:
        return None
    with GPU_LOCK, torch.no_grad():
        tensor = CUSTOM_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        logits = custom_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, top_k)
    results = []
    for i in range(top_k):
        idx = top_idxs[i].item()
        raw = CUSTOM_CLASSES[idx]
        results.append({
            "label": CUSTOM_LABELS_DISPLAY.get(raw, raw),
            "label_en": raw,
            "confidence": float(top_probs[i].item() * 100),
        })
    return results


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
        <div class="header-sub">Nesne Tanima &middot; Gorsel Yorumlama &middot; Yuz Analizi &middot; Sohbet</div>
    </div>

    <!-- Chat -->
    <div class="chat-area" id="chatArea">
        <div class="welcome" id="welcomeMsg">
            <div class="welcome-glass">
                <div class="welcome-icon">&#9889;</div>
                <h2>Merhaba!</h2>
                <p>Benimle sohbet edebilir, fotograf yukleyerek analiz yapabilirsin.<br>
                Gorseldeki nesneleri taniyor, yuzleri analiz ediyor ve gorseli yorumluyorum.<br>
                Fotograf yukleyip soru da sorabilirsin! Ornegin: "Bu nedir, ne ise yarar?"</p>
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
                <input type="text" class="chat-text-input" id="chatInput" placeholder="Mesaj yaz, fotograf yukle veya gorsele soru sor..." autocomplete="off">
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
            chatInput.placeholder = 'Mesaj yaz, fotograf yukle veya gorsele soru sor...';
        }

        function scrollToBottom() {
            chatArea.scrollTop = chatArea.scrollHeight;
        }

        function addUserMessage(imageSrc, userText) {
            welcomeMsg.style.display = 'none';
            const msg = document.createElement('div');
            msg.className = 'msg user';
            var captionText = userText ? userText : 'Analiz istendi';
            msg.innerHTML = `
                <div class="msg-avatar">&#128100;</div>
                <div class="msg-bubble">
                    <img src="${imageSrc}" alt="Fotograf">
                    <div class="caption">${captionText}</div>
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
            // Uzun cevaplarda en alta degil, cevabin BASINA kaydir ki bastan okunabilsin
            scrollToMessageTop(msg);
        }

        function scrollToMessageTop(msg) {
            // Yeni mesajin ust kenarini sohbet alaninin ustune hizala (12px bosluk birak)
            const top = msg.offsetTop - chatArea.offsetTop - 12;
            chatArea.scrollTo({ top: top, behavior: 'smooth' });
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
                data.faces.forEach((face, idx) => {
                    const isMale = face.gender === 'Erkek';
                    const icon = isMale ? '&#128104;' : '&#128105;';
                    var emotionHtml = '';
                    if (data.emotions && data.emotions[idx]) {
                        var emo = data.emotions[idx];
                        var emoIcons = {'mutlu':'&#128522;','uzgun':'&#128546;','kizgin':'&#128544;','saskin':'&#128558;','korkmus':'&#128552;','igrenmis':'&#129326;','notr':'&#128528;','ozguvenli':'&#128526;','yorgun':'&#128564;'};
                        var emoIcon = emoIcons[emo.emotion] || '&#128528;';
                        emotionHtml = '<div class="face-age">' + emoIcon + ' Duygu: ' + emo.emotion + '</div>';
                    }
                    html += `
                    <div class="face-card">
                        <div class="face-icon ${isMale ? 'male' : 'female'}">${icon}</div>
                        <div class="face-detail">
                            <div class="face-gender">${face.gender}</div>
                            <div class="face-age">Tahmini yas: ~${face.age}</div>
                            ${emotionHtml}
                        </div>
                    </div>`;
                });
                sections++;
            }

            // Nesne tanima - top 5
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
                // Top-5 diger tahminler
                if (data.object_top5 && data.object_top5.length > 1) {
                    var others = data.object_top5.slice(1).filter(o => o.confidence >= 10);
                    if (others.length > 0) {
                        html += '<div style="margin-top:8px;font-size:0.78em;color:#5a8ab5;">';
                        html += 'Diger tahminler: ' + others.map(o => o.label + ' (%' + o.confidence.toFixed(1) + ')').join(', ');
                        html += '</div>';
                    }
                }
                sections++;
            }

            // Sahne etiketleri
            if (data.scene_tags && data.scene_tags.length > 0) {
                if (sections > 0) html += '<div class="section-divider"></div>';
                html += `
                <div class="section-title">
                    <div class="section-icon" style="background:rgba(52,152,219,0.12);">&#127749;</div>
                    <div class="section-name">Sahne Ozellikleri</div>
                </div>
                <div style="display:flex;flex-wrap:wrap;gap:6px;">`;
                data.scene_tags.forEach(tag => {
                    html += '<span style="background:rgba(52,152,219,0.1);color:#2471a3;padding:4px 12px;border-radius:20px;font-size:0.82em;font-weight:500;">' + tag + '</span>';
                });
                html += '</div>';
                sections++;
            }

            // Ek analizler (aktivite, stil, mevsim, isik, renk)
            if (data.extra) {
                var tags = [];
                if (data.extra.activities && data.extra.activities.length > 0 && data.extra.activities[0].confidence >= 35)
                    tags.push('&#127939; ' + data.extra.activities[0].label);
                if (data.extra.style && data.extra.style.confidence >= 30)
                    tags.push('&#127912; ' + data.extra.style.label);
                if (data.extra.season && data.extra.season.confidence >= 40)
                    tags.push('&#127807; ' + data.extra.season.label);
                if (data.extra.light && data.extra.light.confidence >= 35)
                    tags.push('&#128161; ' + data.extra.light.label);
                if (data.extra.place && data.extra.place.confidence >= 30)
                    tags.push('&#128205; ' + data.extra.place.label);
                if (data.extra.colors && data.extra.colors.length > 0)
                    tags.push('&#127912; ' + data.extra.colors.map(c => c.label).join(', '));

                if (tags.length > 0) {
                    if (sections > 0) html += '<div class="section-divider"></div>';
                    html += '<div class="section-title"><div class="section-icon" style="background:rgba(52,152,219,0.1);">&#128300;</div><div class="section-name">Detayli Analiz</div></div>';
                    html += '<div style="display:flex;flex-wrap:wrap;gap:6px;">';
                    tags.forEach(tag => {
                        html += '<span style="background:rgba(52,152,219,0.08);color:#2471a3;padding:5px 12px;border-radius:20px;font-size:0.8em;font-weight:500;">' + tag + '</span>';
                    });
                    html += '</div>';
                    sections++;
                }
            }

            // Hicbir anlamli sonuc yoksa bilgi mesaji goster
            if (sections === 0 && !data.local_interpretation && !data.gemini_interpretation) {
                html = `<div class="no-result">Fotografta yeterli guvenle taninan bir oge bulunamadi.</div>`;
            }

            // Kendi modelimizin yorumu (rule-based, her zaman gosterilir)
            if (data.local_interpretation) {
                if (sections > 0) html += '<div class="section-divider"></div>';
                html += `
                <div class="section-title">
                    <div class="section-icon" style="background:rgba(46,134,193,0.15);">&#129302;</div>
                    <div class="section-name">Model Yorumu</div>
                </div>
                <div style="font-size:0.93em;line-height:1.7;color:#1a3c5e;padding:4px 0;">${formatReply(data.local_interpretation)}</div>`;
                sections++;
            }

            // Gemini yorumu (internet varsa ek olarak gosterilir — karsilastirma icin)
            if (data.gemini_interpretation) {
                if (sections > 0) html += '<div class="section-divider"></div>';
                html += `
                <div class="section-title">
                    <div class="section-icon" style="background:rgba(67,233,123,0.15);">&#10024;</div>
                    <div class="section-name">Gemini Yorumu</div>
                </div>
                <div style="font-size:0.93em;line-height:1.7;color:#1a3c5e;padding:4px 0;">${formatReply(data.gemini_interpretation)}</div>`;
            }

            return html;
        }

        function sendImage() {
            if (!selectedFile) return;
            const file = selectedFile;
            const imageSrc = previewThumb.src;
            const userText = chatInput.value.trim();

            addUserMessage(imageSrc, userText);
            clearPreview();
            chatInput.value = '';
            addTyping();

            const formData = new FormData();
            formData.append('file', file);
            formData.append('text', userText);
            formData.append('session_id', sessionId);

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
            // Markdown bold -> HTML bold
            text = text.replace(/\*\*(.+?)\*\*/g, '<strong style="color:#1a5276;">$1</strong>');
            // Markdown italic
            text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
            // Newlines
            text = text.split(String.fromCharCode(10)).join('<br>');
            return text;
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

    # 1) Nesne tanima - CLIP top 5
    clip_results = detect_object(image, top_k=5)
    obj_results = clip_results  # varsayılan: CLIP

    # 2) Yuz analizi (yas & cinsiyet) - yuz varsa yap
    faces = analyze_face(image_array)
    result["faces"] = faces

    # 3) Kişi tanıma - yüz tespit edildiyse çalıştır
    celebrity_info = None
    if faces and face_db is not None:
        label_tr, face_conf = recognize_face(image)
        celebrity_info = {
            "label": label_tr,
            "confidence": round(face_conf, 1),
        }
    result["celebrity"] = celebrity_info

    # 4) Özel eğitilmiş model (sporcu/takım/spor dalı)
    custom_results = detect_custom_object(image, top_k=3)
    if custom_results:
        result["custom_top3"] = [{"label": r["label"], "confidence": round(r["confidence"], 1)} for r in custom_results]

    # 5) ENSEMBLE KARARI — hangi kaynağı "ana nesne" olarak göstereceğiz?
    # Öncelik sırası:
    #   a) face_db celebrity — yüz tanıdıysa ve >=50, en güvenilir
    #   b) Custom model — top1>=50 & margin>=20 & custom > clip
    #   c) CLIP — diğer tüm durumlarda
    clip_top1 = clip_results[0]
    chosen_source = "clip"
    chosen_top = clip_top1

    # (a) celebrity tanındıysa
    _celeb_known = (
        celebrity_info
        and celebrity_info["label"] not in ("Bilinmeyen kişi", "Bilinmeyen kisi", "Yüz tespit edilemedi", "Veritabanı yüklenmedi")
        and celebrity_info["confidence"] >= 50
    )
    # (b) custom model kararlı mı?
    _custom_triggers = False
    if custom_results:
        c1 = custom_results[0]["confidence"]
        c2 = custom_results[1]["confidence"] if len(custom_results) > 1 else 0.0
        if c1 >= 50 and (c1 - c2) >= 20:
            _custom_triggers = True

    if _celeb_known:
        # face_db en güvenilir — her durumda kazanır
        chosen_source = "celebrity"
        chosen_top = {
            "label": celebrity_info["label"],
            "label_en": celebrity_info["label"],
            "confidence": celebrity_info["confidence"],
        }
        obj_results = [chosen_top] + clip_results
    elif clip_top1["confidence"] >= 85:
        # CLIP zaten çok emin (örn. Fenerbahçe forması %99) — dokunma
        pass
    elif _custom_triggers:
        # CLIP emin değil + custom model kararlı → custom kazanır
        chosen_source = "custom"
        chosen_top = custom_results[0]
        obj_results = custom_results + clip_results
    # else: CLIP'te kalıyoruz (chosen_top already clip_top1)

    result["custom_detected"] = (chosen_source == "custom")
    result["source"] = chosen_source  # debug

    # Halusinasyon kontrolu: CLIP'ten gelen dusuk guvenli tahminleri reddet.
    # Custom model ve face_db kendi guven kontrollerini zaten yapiyor.
    #
    # ONEMLI: Esik HAM KOSINUS uzerinden uygulanir, softmax guveni uzerinden DEGIL.
    # Sebep: softmax(sim*100) 961 sinif uzerinde calistigi icin en yakin etikete
    # her zaman yuksek olasilik (%60-90) verir -- listede olmayan bir nesne
    # (orn. mousepad) bile en benzer etikete (orn. "kablosuz sarj cihazi") yuksek
    # softmax guveniyle eslesir ve eski softmax esigini gecerdi. Ham kosinus ise
    # "goruntu bu etikete gercekten benziyor mu?" sorusunu yansitir:
    #   ~0.24+  -> guvenilir dogru eslesme
    #   ~0.20-0.235 -> belirsiz, en yakin tahmin
    #   <0.20   -> listede karsiligi yok, tanimadik
    # Esikler gercek olcumlere gore kalibre edildi (ViT-L/14 openai):
    #   su isiticisi 0.247 / DSLR 0.231 (DOGRU)  >> kus->salata 0.221 (YANLIS) / vasak 0.194 (zayif)
    COSINE_UNKNOWN = 0.18    # altinda: "tanimadim" (listede karsiligi yok)
    COSINE_UNSURE = 0.225    # altinda: "belirsiz, en yakin tahmin"
    chosen_conf = chosen_top["confidence"]
    display_label = chosen_top["label"]
    is_uncertain = False
    if chosen_source == "clip":
        top_cosine = chosen_top.get("cosine", 1.0)
        if top_cosine < COSINE_UNKNOWN:
            display_label = "Bu görseli tanıyamadım"
            is_uncertain = True
        elif top_cosine < COSINE_UNSURE:
            display_label = f"Belirsiz — en yakın tahmin: {chosen_top['label']}"
            is_uncertain = True

    result["object"] = {
        "label": display_label,
        "confidence": round(chosen_conf, 1),
        "uncertain": is_uncertain,
    }
    result["object_top5"] = [{"label": r["label"], "confidence": round(r["confidence"], 1)} for r in obj_results[:5]]

    # 4) Görsel sahne/kategori tespiti (CLIP ile)
    scene_labels_tr = ["ic mekan", "dis mekan", "doga", "sehir", "deniz", "dag", "orman",
                       "gece", "gunduz", "gun batimi", "gun dogumu", "yagmurlu", "karli", "gunesli",
                       "stüdyo cekimi", "selfie", "grup fotografi", "manzara", "yakin cekim", "havadan cekim"]
    scene_labels_en = ["indoor", "outdoor", "nature", "city", "sea", "mountain", "forest",
                       "night", "daytime", "sunset", "sunrise", "rainy", "snowy", "sunny",
                       "studio shot", "selfie", "group photo", "landscape", "close-up", "aerial view"]
    with GPU_LOCK, torch.no_grad():
        scene_prompts = [f"a photo taken {en}" if en in ("indoor", "outdoor") else f"a {en} photo" for en in scene_labels_en]
        scene_tokens = clip_tokenizer(scene_prompts).to(CLIP_DEVICE)
        scene_features = clip_model.encode_text(scene_tokens)
        scene_features = scene_features / scene_features.norm(dim=-1, keepdim=True)
        img_tensor = clip_preprocess(image).unsqueeze(0).to(CLIP_DEVICE, dtype=CLIP_DTYPE)
        img_features = clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        scene_sim = (img_features @ scene_features.T)[0]
        scene_probs = torch.softmax(scene_sim * 100, dim=0)
    top3_scene_probs, top3_scene_idx = torch.topk(scene_probs, 3)
    scene_tags = []
    for i in range(3):
        if top3_scene_probs[i].item() > 0.35:
            scene_tags.append(scene_labels_tr[top3_scene_idx[i].item()])
    result["scene_tags"] = scene_tags

    # 5) Duygu tespiti (yüz varsa)
    emotion_results = []
    if faces:
        emotion_labels_en = ["happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral", "confident", "tired"]
        emotion_labels_tr = ["mutlu", "uzgun", "kizgin", "saskin", "korkmus", "igrenmis", "notr", "ozguvenli", "yorgun"]
        img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        faces_detected = face_app.get(img_bgr)
        h, w = image_array.shape[:2]
        for face in faces_detected:
            bbox = face.bbox.astype(int)
            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w, bbox[2]), min(h, bbox[3])
            face_crop = image_array[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_pil = Image.fromarray(face_crop)
            with GPU_LOCK, torch.no_grad():
                face_tensor = clip_preprocess(face_pil).unsqueeze(0).to(CLIP_DEVICE, dtype=CLIP_DTYPE)
                face_features = clip_model.encode_image(face_tensor)
                face_features = face_features / face_features.norm(dim=-1, keepdim=True)
                emo_prompts = [f"a photo of a {e} person" for e in emotion_labels_en]
                emo_tokens = clip_tokenizer(emo_prompts).to(CLIP_DEVICE)
                emo_features = clip_model.encode_text(emo_tokens)
                emo_features = emo_features / emo_features.norm(dim=-1, keepdim=True)
                emo_sim = (face_features @ emo_features.T)[0]
                emo_probs = torch.softmax(emo_sim * 100, dim=0)
            top_emo_prob, top_emo_idx = torch.topk(emo_probs, 1)
            emotion_results.append({
                "emotion": emotion_labels_tr[top_emo_idx[0].item()],
                "confidence": round(float(top_emo_prob[0].item() * 100), 1)
            })
    result["emotions"] = emotion_results

    # 6) Ek CLIP analizleri (renk, aktivite, stil, mevsim, ışık, mekan)
    extra = analyze_image_extra(image)

    # --- TUTARLILIK FİLTRELERİ (hem çipler hem yorum için) ---
    # İnsan gerektiren sahneler yüz yoksa at
    _people_scenes = {"selfie", "grup fotografi"}
    if not faces:
        scene_tags = [s for s in scene_tags if s not in _people_scenes]
        result["scene_tags"] = scene_tags

    # Yemek nesnesi ve insan aktivite tutarlılık kontrolü
    _food_keywords = {"elma", "armut", "portakal", "muz", "çilek", "karpuz", "üzüm",
                      "domates", "biber", "mantar", "pizza", "hamburger", "makarna",
                      "ekmek", "pasta", "kek", "kurabiye", "dondurma", "çorba",
                      "salata", "yemek", "meyve", "sebze", "tatlı", "peynir"}
    main_obj_label = obj_results[0]["label"].lower() if obj_results else ""
    is_food_object = any(kw in main_obj_label for kw in _food_keywords)

    _human_activities = {"yemek yeme", "içecek içme", "yürüme", "koşma", "oturma",
                         "ayakta durma", "dans etme", "spor yapma", "okuma",
                         "çalışma", "yemek pişirme", "uyuma", "konuşma", "gülme",
                         "araç kullanma", "yüzme", "fotoğraf çekme", "alışveriş yapma",
                         "ders çalışma", "resim yapma", "müzik çalma", "egzersiz yapma"}
    filtered_activities = []
    for act in extra.get("activities", []):
        name = act.get("label", "")
        # Yüz yoksa insan aktivitesi atılır
        if name in _human_activities and not faces:
            continue
        # Yemek nesnesinde "yemek yeme" redundant
        if is_food_object and name in ("yemek yeme", "yemek pişirme"):
            continue
        # Düşük güveni at (%55 altı)
        if act.get("confidence", 0) < 55:
            continue
        filtered_activities.append(act)
    extra["activities"] = filtered_activities

    # Mevsim ancak dış mekan sahnesinde mantıklı
    _outdoor_scenes = {"dis mekan", "doga", "sehir", "deniz", "dag", "orman",
                       "gece", "gun batimi", "gun dogumu", "manzara", "havadan cekim",
                       "karli", "yagmurlu", "gunesli", "gunduz"}
    has_outdoor = any(s in _outdoor_scenes for s in scene_tags)
    if not has_outdoor:
        extra["season"] = None

    result["extra"] = {
        "colors": extra.get("colors", []),
        "style": extra.get("style"),
        "season": extra.get("season"),
        "light": extra.get("light"),
        "place": extra.get("place"),
        "activities": extra.get("activities", []),
    }

    # 7) Kendi modelimizin yorumu (her zaman çalışır, internet gerekmez)
    user_text = request.form.get("text", "").strip()
    session_id = request.form.get("session_id", "default")

    local_interpretation = generate_offline_interpretation(
        obj_results, faces, emotion_results, scene_tags,
        celebrity=result.get("celebrity"), user_text=user_text, extra=extra,
        object_uncertain=is_uncertain
    )
    result["local_interpretation"] = local_interpretation

    # Zengin bağlam — Gemini yorumlaması için kullanılır
    context_parts = []
    obj_str = ", ".join([f"{r['label']} (%{r['confidence']:.0f})" for r in obj_results[:5] if r['confidence'] >= 5])
    if obj_str:
        context_parts.append(f"Nesne tanima (top-5): {obj_str}")
    if is_uncertain:
        context_parts.append(
            "UYARI: Yerel nesne tanima modeli bu gorselden emin degil (dusuk guven). "
            "Yukaridaki tahminlere fazla guvenme; gorsele kendi gozunle bakarak yorum yap. "
            "Eger emin olamiyorsan 'tam olarak anlayamadim' demekten cekinme."
        )
    if scene_tags:
        context_parts.append(f"Sahne ozellikleri: {', '.join(scene_tags)}")
    if faces:
        for i, face in enumerate(faces):
            face_info = f"Yuz {i+1}: {face['gender']}, ~{face['age']} yas"
            if i < len(emotion_results):
                face_info += f", duygu: {emotion_results[i]['emotion']}"
            context_parts.append(face_info)
    if result.get("celebrity") and result["celebrity"]["label"] not in ("Bilinmeyen kişi", "Bilinmeyen kisi", "Yüz tespit edilemedi"):
        context_parts.append(f"Kisi tanima: {result['celebrity']['label']}")
    if extra.get("colors"):
        context_parts.append(f"Baskın renkler: {', '.join([c['label'] for c in extra['colors'][:3]])}")
    if extra.get("activities") and extra["activities"][0]["confidence"] >= 35:
        context_parts.append(f"Aktivite: {extra['activities'][0]['label']}")
    if extra.get("style"):
        context_parts.append(f"Fotograf stili: {extra['style']['label']}")
    if extra.get("season"):
        context_parts.append(f"Mevsim: {extra['season']['label']}")
    if extra.get("place"):
        context_parts.append(f"Mekan: {extra['place']['label']}")
    context_str = "\n".join(context_parts) if context_parts else "Ek bilgi yok"

    # 7) Gemini ile zengin yorumlama (internet varsa ek olarak çalışır)
    gemini_interpretation = ""
    if gemini_client:
        try:
            if user_text:
                gemini_prompt = f"""Kullanici bu gorseli gonderdi ve su soruyu sordu: "{user_text}"

Yardimci olmasi icin yerel yapay zeka modellerimizin tahminleri (HATALI olabilir, sadece ipucu):
{context_str}

ONCE goruntuye kendi gozunle dikkatlice bak ve adim adim dusun:
- Gorselde gercekten ne var? Yukaridaki tahminler gordugunle celisiyorsa KENDI gozlemine guven.
- Detaylardan ne cikarim yapabilirsin? (nerede cekilmis, ne zaman, ne oluyor, kisiler ne yapiyor)
SONRA bu akil yurutmeye dayanarak kullanicinin sorusunu detayli, isabetli ve samimi bir Turkce ile cevapla.
Gorseldeki nesnelerin ne ise yaradigini ve sahnenin ne anlattigini da acikla."""
            else:
                gemini_prompt = f"""Bu gorseli kapsamli ve derin bir sekilde yorumla.

Yardimci olmasi icin yerel yapay zeka modellerimizin tahminleri (HATALI olabilir, sadece ipucu):
{context_str}

ONCE goruntuye kendi gozunle dikkatlice bak: Yukaridaki tahminler gordugunle celisiyorsa KENDI gozlemine guven.
Sonra adim adim dusun ve gozlemlerinden CIKARIM yap: Sadece "ne var" deme, "bu ne anlama geliyor, neden boyle, ne oluyor" diye yorumla.

Asagidaki basliklar altinda gorseli degerlendir:

1. **Sahne Tanimi**: Gorselde ne goruyorsun? Ortami, mekani, isigi ve cekildigi yeri/zamani tarif et.
2. **Detay Analizi**: Onemli nesneleri/kisileri, bunlarin ne ise yaradigini ve detaylardan cikardigin ipuclarini acikla.
3. **Atmosfer & Duygu**: Gorsel nasil bir his uyandiriyor ve bunu hangi detaylardan anliyorsun?
4. **Ilginc Detay / Cikarim**: Dikkat ceken bir sey veya goruntuden yaptigin akilli bir cikarim.

Turkce yaz, dogal ve akici bir dil kullan, teknik jargondan kacin. Her baslik 2-3 cumle, akil yurutmeni yansitan dolu yorumlar olsun."""

            img_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            text_part = types.Part.from_text(text=gemini_prompt)

            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[types.Content(role="user", parts=[img_part, text_part])],
                config=types.GenerateContentConfig(
                    system_instruction=GEMINI_SYSTEM_PROMPT,
                    max_output_tokens=1400,
                    temperature=0.7,
                    # Dinamik dusunme: Gemini gorseli analiz etmeden once akil yurutur
                    # (-1 = model ne kadar dusunecegine kendi karar verir). Yorum derinligini artirir.
                    thinking_config=types.ThinkingConfig(thinking_budget=-1)
                )
            )
            try:
                gemini_interpretation = response.text
            except Exception:
                gemini_interpretation = ""

            if session_id and gemini_interpretation:
                user_msg = user_text if user_text else "Gorsel analiz istendi"
                _add_to_history(session_id, f"[Gorsel yuklendi] {user_msg}", gemini_interpretation)
        except Exception as e:
            gemini_interpretation = ""

    result["gemini_interpretation"] = gemini_interpretation

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
