"""
SADECE yeni seçilen sınıfları indir ve mevcut custom_data'ya EKLE.
Eski 10 sınıfa dokunmaz. scrape_data.py'nin fonksiyonlarını yeniden kullanır.
Kullanım: python scrape_new.py
"""
import os, shutil, time
import scrape_data as s

YENI_SINIFLAR = [
    "realmadrid", "barcelona", "bayernmunich", "manchester_city", "liverpool",
    "haaland", "mbappe", "neymar", "vinicius", "bellingham",
]

# Eski 10 sınıf IMAGES_PER_CLASS=300 ile toplanmış (~120 geçerli). Aynısını hedefliyoruz.
HEDEF = 300  # 4 sorguya bölününce ~75/sorgu -> filtreden sonra ~120 kalır

def _temizle(cls):
    """Bu sınıfın eski (ince) train/val klasörlerini sil — temiz yeniden indirme."""
    for d in (os.path.join(s.TRAIN_DIR, cls), os.path.join(s.VAL_DIR, cls)):
        shutil.rmtree(d, ignore_errors=True)


def download_images_fixed(class_name, search_queries, per_query):
    """Her sorguyu AYRI alt klasöre indirir (icrawler dosya ezme bug'ını önler),
    sonra hepsini tek bir temp klasörde birleştirir."""
    from icrawler.builtin import BingImageCrawler
    base = os.path.join(s.DATA_DIR, "temp", class_name)
    merged = os.path.join(base, "_merged")
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(merged, exist_ok=True)

    counter = 0
    for qi, query in enumerate(search_queries):
        qdir = os.path.join(base, f"q{qi}")
        os.makedirs(qdir, exist_ok=True)
        print(f"  Araniyor: '{query}' ({per_query} gorsel)", flush=True)
        crawler = BingImageCrawler(storage={"root_dir": qdir},
                                   feeder_threads=1, parser_threads=1,
                                   downloader_threads=4)
        crawler.crawl(keyword=query, max_num=per_query, min_size=(100, 100))
        # bu sorgunun dosyalarını merged'e benzersiz isimle taşı
        for f in os.listdir(qdir):
            fp = os.path.join(qdir, f)
            if os.path.isfile(fp):
                ext = os.path.splitext(f)[1] or ".jpg"
                shutil.move(fp, os.path.join(merged, f"{counter:05d}{ext}"))
                counter += 1
    return merged

def main():
    print("=" * 60)
    print(f"  YENİ {len(YENI_SINIFLAR)} SINIF İNDİRİLİYOR (mevcut veriye eklenecek)")
    print("=" * 60)
    temp_root = os.path.join(s.DATA_DIR, "temp")
    shutil.rmtree(temp_root, ignore_errors=True)

    ozet = []
    for ci, cls in enumerate(YENI_SINIFLAR, 1):
        queries = s.CLASSES[cls]
        print(f"\n[{ci}/{len(YENI_SINIFLAR)}] {cls.upper()} indiriliyor...", flush=True)
        t0 = time.time()
        _temizle(cls)  # eski ince veriyi sil
        # her sorgudan ~100 iste; 4 sorgu birikince ezme olmadan 120+ kalir
        temp_dir = download_images_fixed(cls, queries, 100)
        valid = s.filter_images(temp_dir)
        n_train, n_val = s.split_train_val(valid, cls)
        ozet.append((cls, n_train, n_val))
        print(f"  -> {cls}: train={n_train} val={n_val} ({time.time()-t0:.0f} sn)", flush=True)

    shutil.rmtree(temp_root, ignore_errors=True)

    print("\n" + "=" * 60)
    print("  ÖZET — yeni eklenen sınıflar")
    print("=" * 60)
    tt = tv = 0
    for cls, nt, nv in ozet:
        print(f"  {cls:18s} train={nt:4d}  val={nv:3d}")
        tt += nt; tv += nv
    print(f"\n  YENİ TOPLAM: {tt} train + {tv} val = {tt+tv} görsel")
    print(f"  custom_data artık {10+len(YENI_SINIFLAR)} sınıflı.")
    print("  Sonraki adım: python train_custom.py  (model 20 sınıfta yeniden eğitilir)")

if __name__ == "__main__":
    main()
