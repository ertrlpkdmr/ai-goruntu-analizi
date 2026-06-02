"""
Mevcut 10 sinif icin DAHA COK ve TEMIZ veri toplama.
- Sadece modelin bildigi 10 sinif (custom_classes.json ile ayni)
- Zengin/cesitli sorgular (zayif siniflara ozel)
- messi/ronaldo icin "tek kisi" odakli sorgular (cift-kisi kirliligini azaltir)
- dHash ile yinelenen (duplicate) eleme: hem mevcut veriyle hem kendi icinde
- Yeni gorseller SADECE train'e eklenir; val sabit/temiz kalir (sizinti yok)

Kullanim:
  python scrape_more_data.py              # tam calistirma
  python scrape_more_data.py --test       # her sinif 1 sorgu, hizli deneme
"""

import os
import sys
import shutil
import numpy as np
from PIL import Image
from icrawler.builtin import BingImageCrawler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "custom_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
STAGING_DIR = os.path.join(DATA_DIR, "_staging")

TARGET_PER_CLASS = 200        # sinif basina hedeflenen TOPLAM (mevcut + yeni) train gorsel
MIN_FILE_SIZE = 10 * 1024     # 10KB
MIN_DIM = 100                 # kenar < 100px ise at
DHASH_THRESHOLD = 6           # bu Hamming mesafesinin altindakiler "ayni" sayilir

# Zengin sorgu listeleri. messi/ronaldo TEK KISI odakli (cift-kisi karelerinden kacin).
CLASSES = {
    "galatasaray": [
        "Galatasaray 2024 mac", "Galatasaray forma", "Galatasaray taraftar tribun",
        "Galatasaray stadyum", "Galatasaray gol sevinci", "Galatasaray sari kirmizi",
    ],
    "fenerbahce": [
        "Fenerbahce 2024 mac", "Fenerbahce forma", "Fenerbahce taraftar tribun",
        "Fenerbahce stadyum Kadikoy", "Fenerbahce gol sevinci", "Fenerbahce sari lacivert",
    ],
    "besiktas": [
        "Besiktas 2024 mac", "Besiktas forma", "Besiktas taraftar tribun",
        "Besiktas Vodafone Park", "Besiktas gol sevinci", "Besiktas siyah beyaz",
    ],
    "trabzonspor": [
        "Trabzonspor 2024 mac", "Trabzonspor forma bordo mavi", "Trabzonspor taraftar",
        "Trabzonspor stadyum", "Trabzonspor gol sevinci", "Trabzonspor kupa",
        "Trabzonspor oyuncular", "Trabzonspor antrenman",
    ],
    "messi": [
        "Lionel Messi Inter Miami 2024", "Messi Argentina milli takim",
        "Messi yakin cekim portre", "Messi gol sevinci tek",
        "Messi pembe forma Inter Miami", "Messi dribbling top",
    ],
    "ronaldo": [
        "Cristiano Ronaldo Al Nassr 2024", "Ronaldo Portekiz milli takim",
        "Ronaldo yakin cekim portre", "Ronaldo siuu gol sevinci tek",
        "Ronaldo sari forma Al Nassr", "Ronaldo sut cekisi",
    ],
    "futbol": [
        "futbol mac sahasi", "futbol topu cim", "futbol oyuncu sut",
        "soccer match stadium", "futbol kale gol", "futbol koltuk korner",
    ],
    "basketbol": [
        "NBA basketbol mac", "basketbol pota smac dunk", "basketbol dribbling oyuncu",
        "Euroleague basketbol", "basketbol salon parke", "basketbol sut three point",
    ],
    "tenis": [
        "tenis mac kort", "tennis player forehand", "tenis raketi vurus",
        "Wimbledon tenis", "tenis servis atisi", "tenis toprak kort",
    ],
    "voleybol": [
        "voleybol mac salon", "volleyball spike smac", "voleybol blok file",
        "voleybol takim kutlama", "voleybol servis", "voleybol VNL",
    ],
}


def dhash(pil_img, hash_size=8):
    """8x8 dHash -> 64 bitlik numpy bool dizisi."""
    img = pil_img.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.int16)
    diff = a[:, 1:] > a[:, :-1]
    return diff.flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def load_existing_hashes():
    """train + val'deki TUM gorsellerin dHash'lerini topla (sizinti & cross-class dedup)."""
    hashes = []
    for root in (TRAIN_DIR, VAL_DIR):
        if not os.path.isdir(root):
            continue
        for cls in os.listdir(root):
            cdir = os.path.join(root, cls)
            if not os.path.isdir(cdir):
                continue
            for fn in os.listdir(cdir):
                fp = os.path.join(cdir, fn)
                try:
                    with Image.open(fp) as im:
                        hashes.append(dhash(im.convert("RGB")))
                except Exception:
                    pass
    return hashes


def is_duplicate(h, hash_list):
    for existing in hash_list:
        if hamming(h, existing) <= DHASH_THRESHOLD:
            return True
    return False


def download_class(class_name, queries, per_query):
    temp_dir = os.path.join(STAGING_DIR, class_name)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    for q in queries:
        print(f"    Araniyor: '{q}' (~{per_query})")
        try:
            crawler = BingImageCrawler(
                storage={"root_dir": temp_dir},
                feeder_threads=1, parser_threads=2, downloader_threads=4,
            )
            crawler.crawl(keyword=q, max_num=per_query, min_size=(MIN_DIM, MIN_DIM))
        except Exception as e:
            print(f"      ! sorgu hatasi: {e}")
    return temp_dir


def next_train_index(class_dir):
    """Mevcut dosyalari ezmemek icin bir sonraki index'i bul."""
    mx = -1
    if os.path.isdir(class_dir):
        for fn in os.listdir(class_dir):
            base = os.path.splitext(fn)[0]
            num = base.split("_")[-1]
            if num.isdigit():
                mx = max(mx, int(num))
    return mx + 1


def main():
    test_mode = "--test" in sys.argv
    print("=" * 60)
    print("  MEVCUT 10 SINIF - VERI ARTIRMA" + ("  [TEST MODU]" if test_mode else ""))
    print("=" * 60)

    if not os.path.isdir(TRAIN_DIR):
        print("HATA: custom_data/train bulunamadi.")
        return

    print("Mevcut gorsellerin parmak izleri (dHash) cikariliyor...")
    existing = load_existing_hashes()
    print(f"  {len(existing)} mevcut gorsel tarandi.\n")

    total_added = 0
    summary = []

    for class_name, queries in CLASSES.items():
        if test_mode:
            queries = queries[:1]
        print(f"[{class_name.upper()}] indiriliyor...")

        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        have = len([f for f in os.listdir(train_class_dir)
                    if os.path.isfile(os.path.join(train_class_dir, f))])
        need = max(0, TARGET_PER_CLASS - have)
        if test_mode:
            need = 5
        if need == 0:
            print(f"  Zaten {have} var, hedefe ulasilmis. Atlaniyor.\n")
            summary.append((class_name, have, 0))
            continue

        # biraz fazla indir (eleme sonrasi need'e ulasmak icin)
        per_query = max(15, (need * 3) // len(queries) + 1)
        temp_dir = download_class(class_name, queries, per_query)

        # filtrele + dedup + train'e ekle
        accepted = list(existing)  # mevcutlar + bu kosuda kabul edilenler birlikte
        idx = next_train_index(train_class_dir)
        added = 0
        files = sorted(os.listdir(temp_dir)) if os.path.isdir(temp_dir) else []
        for fn in files:
            if added >= need:
                break
            fp = os.path.join(temp_dir, fn)
            if not os.path.isfile(fp) or os.path.getsize(fp) < MIN_FILE_SIZE:
                continue
            try:
                with Image.open(fp) as im:
                    im.verify()
                with Image.open(fp) as im:
                    rgb = im.convert("RGB")
                    w, h = rgb.size
                    if w < MIN_DIM or h < MIN_DIM:
                        continue
                    hsh = dhash(rgb)
                    if is_duplicate(hsh, accepted):
                        continue
                    accepted.append(hsh)
                    dst = os.path.join(train_class_dir, f"{class_name}_{idx:04d}.jpg")
                    rgb.save(dst, "JPEG", quality=92)
                    idx += 1
                    added += 1
            except Exception:
                continue

        existing = accepted  # sonraki siniflar da bunlara karsi dedup edilsin
        total_added += added
        new_total = have + added
        summary.append((class_name, new_total, added))
        print(f"  +{added} yeni (toplam train: {new_total})\n")

    # staging temizle
    if os.path.exists(STAGING_DIR):
        shutil.rmtree(STAGING_DIR)

    print("=" * 60)
    print("  OZET (sinif | yeni train toplam | eklenen)")
    for name, tot, add in summary:
        print(f"    {name:<13} {tot:<4} (+{add})")
    print(f"  TOPLAM EKLENEN: {total_added}")
    print("=" * 60)
    print("Sonraki adim:  python train_custom.py")


if __name__ == "__main__":
    main()
