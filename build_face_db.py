"""
Futbolcu Yüz Tanıma Veritabanı Oluşturucu
- Her futbolcu için referans fotoğrafları indirir
- InsightFace ile yüz embedding'lerini çıkarır
- Veritabanını face_db.pkl olarak kaydeder
"""

import os
import pickle
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from icrawler.builtin import BingImageCrawler
from PIL import Image
import shutil

# ===== AYARLAR =====
IMAGES_PER_PERSON = 15  # Her kişi için indirilecek fotoğraf sayısı
MIN_EMBEDDINGS = 3      # Minimum embedding sayısı (bundan az olursa uyarı verir)
FACE_DB_PATH = "face_db.pkl"
TEMP_DIR = "face_ref_temp"

# ===== FUTBOLCU LİSTESİ =====
# Anahtar: klasör adı, Değer: (Arama terimi, Görünen isim)
FOOTBALLERS = {
    # Türk futbolcular
    "ardaguler": ("Arda Güler Real Madrid footballer face", "Arda Güler"),
    "calhanoglu": ("Hakan Çalhanoğlu Inter Milan footballer face", "Hakan Çalhanoğlu"),
    "yildiz": ("Kenan Yıldız Juventus footballer face", "Kenan Yıldız"),
    "cenk_tosun": ("Cenk Tosun Beşiktaş footballer face", "Cenk Tosun"),
    "irfancan": ("İrfan Can Kahveci Fenerbahçe footballer face", "İrfan Can Kahveci"),
    "cengiz_under": ("Cengiz Ünder footballer face", "Cengiz Ünder"),
    "ugurcan": ("Uğurcan Çakır Trabzonspor goalkeeper face", "Uğurcan Çakır"),
    "arda_turan": ("Arda Turan footballer face", "Arda Turan"),
    "burak_yilmaz": ("Burak Yılmaz footballer face", "Burak Yılmaz"),
    "hakansukur": ("Hakan Şükür footballer face", "Hakan Şükür"),

    # İspanya / La Liga
    "messi": ("Lionel Messi footballer face close up", "Lionel Messi"),
    "mbappe": ("Kylian Mbappé Real Madrid footballer face", "Kylian Mbappé"),
    "vinicius": ("Vinicius Jr Real Madrid footballer face", "Vinícius Jr"),
    "bellingham": ("Jude Bellingham Real Madrid footballer face", "Jude Bellingham"),
    "lewandowski": ("Robert Lewandowski Barcelona footballer face", "Robert Lewandowski"),
    "pedri": ("Pedri Barcelona footballer face", "Pedri"),
    "gavi": ("Gavi Barcelona footballer face", "Gavi"),
    "yamal": ("Lamine Yamal Barcelona footballer face", "Lamine Yamal"),
    "rodrygo": ("Rodrygo Real Madrid footballer face", "Rodrygo"),
    "modric": ("Luka Modrić Real Madrid footballer face", "Luka Modrić"),

    # İngiltere / Premier League
    "haaland": ("Erling Haaland Manchester City footballer face", "Erling Haaland"),
    "salah": ("Mohamed Salah Liverpool footballer face", "Mohamed Salah"),
    "debruyne": ("Kevin De Bruyne Manchester City footballer face", "Kevin De Bruyne"),
    "saka": ("Bukayo Saka Arsenal footballer face", "Bukayo Saka"),
    "odegaard": ("Martin Ødegaard Arsenal footballer face", "Martin Ødegaard"),
    "palmer": ("Cole Palmer Chelsea footballer face", "Cole Palmer"),
    "rice": ("Declan Rice Arsenal footballer face", "Declan Rice"),
    "bruno": ("Bruno Fernandes Manchester United footballer face", "Bruno Fernandes"),
    "sonheungmin": ("Son Heung-min Tottenham footballer face", "Son Heung-min"),
    "foden": ("Phil Foden Manchester City footballer face", "Phil Foden"),
    "rashford": ("Marcus Rashford Manchester United footballer face", "Marcus Rashford"),
    "alexander_arnold": ("Trent Alexander-Arnold Liverpool footballer face", "Trent Alexander-Arnold"),
    "kane": ("Harry Kane Bayern Munich footballer face", "Harry Kane"),

    # Almanya / Bundesliga
    "musiala": ("Jamal Musiala Bayern Munich footballer face", "Jamal Musiala"),
    "sane": ("Leroy Sané Bayern Munich footballer face", "Leroy Sané"),
    "wirtz": ("Florian Wirtz Leverkusen footballer face", "Florian Wirtz"),
    "brandt": ("Julian Brandt Dortmund footballer face", "Julian Brandt"),

    # İtalya / Serie A
    "osimhen": ("Victor Osimhen footballer face", "Victor Osimhen"),
    "lautaro": ("Lautaro Martínez Inter Milan footballer face", "Lautaro Martínez"),
    "leao": ("Rafael Leão AC Milan footballer face", "Rafael Leão"),
    "chiesa": ("Federico Chiesa footballer face", "Federico Chiesa"),
    "vlahovic": ("Dušan Vlahović Juventus footballer face", "Dušan Vlahović"),

    # Fransa / Ligue 1
    "dembele": ("Ousmane Dembélé PSG footballer face", "Ousmane Dembélé"),
    "hakimi": ("Achraf Hakimi PSG footballer face", "Achraf Hakimi"),

    # Türkiye Süper Lig yabancılar
    "icardi": ("Mauro Icardi Galatasaray footballer face", "Mauro Icardi"),
    "dzeko": ("Edin Džeko Fenerbahçe footballer face", "Edin Džeko"),
    "talisca": ("Anderson Talisca footballer face", "Anderson Talisca"),
    "gedson": ("Gedson Fernandes Beşiktaş footballer face", "Gedson Fernandes"),

    # Efsaneler
    "ronaldo": ("Cristiano Ronaldo footballer face close up", "Cristiano Ronaldo"),
    "neymar": ("Neymar Jr footballer face close up", "Neymar Jr"),
    "maradona": ("Diego Maradona footballer face", "Diego Maradona"),
    "pele": ("Pelé footballer face", "Pelé"),
    "zidane": ("Zinedine Zidane face", "Zinedine Zidane"),
    "ronaldinho": ("Ronaldinho footballer face", "Ronaldinho"),
    "beckham": ("David Beckham face", "David Beckham"),
    "henry": ("Thierry Henry footballer face", "Thierry Henry"),
    "rooney": ("Wayne Rooney face", "Wayne Rooney"),
    "ibrahimovic": ("Zlatan Ibrahimović face", "Zlatan Ibrahimović"),
}


def download_images():
    """Her futbolcu için referans fotoğrafları indir"""
    os.makedirs(TEMP_DIR, exist_ok=True)

    for key, (search_term, display_name) in FOOTBALLERS.items():
        person_dir = os.path.join(TEMP_DIR, key)
        if os.path.exists(person_dir) and len(os.listdir(person_dir)) >= MIN_EMBEDDINGS:
            print(f"  [ATLA] {display_name} - zaten {len(os.listdir(person_dir))} fotoğraf var")
            continue

        os.makedirs(person_dir, exist_ok=True)
        print(f"  [İNDİR] {display_name} ({search_term})...")

        crawler = BingImageCrawler(
            storage={"root_dir": person_dir},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=2,
        )
        crawler.crawl(keyword=search_term, max_num=IMAGES_PER_PERSON)

    print(f"\nToplam {len(FOOTBALLERS)} futbolcu için fotoğraflar indirildi.")


def build_database():
    """İndirilen fotoğraflardan yüz embedding veritabanı oluştur"""
    print("\nInsightFace modeli yükleniyor...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    face_app.prepare(ctx_id=-1, det_size=(640, 640))
    print("InsightFace hazır!\n")

    face_db = {}  # {key: {"name": str, "embeddings": [np.array, ...]}}
    stats = {"success": 0, "no_face": 0, "error": 0}

    for key, (_, display_name) in FOOTBALLERS.items():
        person_dir = os.path.join(TEMP_DIR, key)
        if not os.path.exists(person_dir):
            print(f"  [UYARI] {display_name} - fotoğraf klasörü bulunamadı")
            continue

        embeddings = []
        files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

        for fname in files:
            fpath = os.path.join(person_dir, fname)
            try:
                img = cv2.imread(fpath)
                if img is None:
                    continue

                faces = face_app.get(img)
                if len(faces) == 0:
                    stats["no_face"] += 1
                    continue

                # En büyük yüzü al (en yakın = ana kişi)
                largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                embeddings.append(largest_face.normed_embedding)
                stats["success"] += 1

            except Exception as e:
                stats["error"] += 1
                continue

        if len(embeddings) >= 1:
            face_db[key] = {
                "name": display_name,
                "embeddings": embeddings
            }
            status = "OK" if len(embeddings) >= MIN_EMBEDDINGS else "AZ"
            print(f"  [{status}] {display_name}: {len(embeddings)} embedding")
        else:
            print(f"  [BASARISIZ] {display_name}: hiç yüz bulunamadı!")

    # Kaydet
    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

    print(f"\n{'='*50}")
    print(f"Veritabanı kaydedildi: {FACE_DB_PATH}")
    print(f"Toplam kişi: {len(face_db)}")
    print(f"Toplam embedding: {sum(len(v['embeddings']) for v in face_db.values())}")
    print(f"Başarılı yüz: {stats['success']}, Yüz bulunamayan: {stats['no_face']}, Hata: {stats['error']}")
    print(f"{'='*50}")

    return face_db


if __name__ == "__main__":
    print("=" * 50)
    print("  FUTBOLCU YÜZ TANIMA VERİTABANI OLUŞTURUCU")
    print("=" * 50)

    print(f"\n1) Referans fotoğrafları indiriliyor ({len(FOOTBALLERS)} futbolcu)...\n")
    download_images()

    print("\n2) Yüz embedding veritabanı oluşturuluyor...\n")
    face_db = build_database()

    print("\n3) Temizlik...")
    # Temp klasörünü silmiyoruz, tekrar kullanılabilir
    print(f"   Referans fotoğrafları '{TEMP_DIR}/' klasöründe saklandı.")
    print("\nBitti! Şimdi app.py'yi başlatabilirsiniz.")
