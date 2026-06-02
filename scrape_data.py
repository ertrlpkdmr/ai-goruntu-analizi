"""
Ozel Nesne Tanima - Otomatik Gorsel Toplama
icrawler kutuphanesi ile Bing/Google gorsel arama
Kullanim: python scrape_data.py
"""

import os
import shutil
import random
from PIL import Image
from icrawler.builtin import BingImageCrawler

# =============================================
# AYARLAR
# =============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "custom_data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

IMAGES_PER_CLASS = 300
TRAIN_RATIO = 0.8
MIN_FILE_SIZE = 10 * 1024  # 10KB

# Sinif isimleri ve arama terimleri
CLASSES = {
    # ==============================
    # TURKIYE SUPER LIG TAKIMLARI
    # ==============================
    "galatasaray": ["Galatasaray forma", "Galatasaray logo", "Galatasaray mac", "Galatasaray futbol"],
    "fenerbahce": ["Fenerbahce forma", "Fenerbahce logo", "Fenerbahce mac", "Fenerbahce futbol"],
    "besiktas": ["Besiktas forma", "Besiktas logo", "Besiktas mac", "Besiktas futbol"],
    "trabzonspor": ["Trabzonspor forma", "Trabzonspor logo", "Trabzonspor mac", "Trabzonspor futbol"],
    "basaksehir": ["Basaksehir forma", "Istanbul Basaksehir mac", "Basaksehir futbol", "Basaksehir logo"],
    "adanademirspor": ["Adana Demirspor forma", "Adana Demirspor mac", "Adana Demirspor futbol"],
    "antalyaspor": ["Antalyaspor forma", "Antalyaspor mac", "Antalyaspor futbol", "Antalyaspor logo"],
    "alanyaspor": ["Alanyaspor forma", "Alanyaspor mac", "Alanyaspor futbol", "Alanyaspor logo"],
    "konyaspor": ["Konyaspor forma", "Konyaspor mac", "Konyaspor futbol", "Konyaspor logo"],
    "sivasspor": ["Sivasspor forma", "Sivasspor mac", "Sivasspor futbol", "Sivasspor logo"],
    "kasimpasa": ["Kasimpasa forma", "Kasimpasa mac", "Kasimpasa futbol", "Kasimpasa logo"],
    "kayserispor": ["Kayserispor forma", "Kayserispor mac", "Kayserispor futbol", "Kayserispor logo"],
    "rizespor": ["Rizespor forma", "Caykur Rizespor mac", "Rizespor futbol", "Rizespor logo"],
    "samsunspor": ["Samsunspor forma", "Samsunspor mac", "Samsunspor futbol", "Samsunspor logo"],
    "gaziantepfk": ["Gaziantep FK forma", "Gaziantep FK mac", "Gaziantep FK futbol", "Gaziantep FK logo"],
    "hatayspor": ["Hatayspor forma", "Hatayspor mac", "Hatayspor futbol", "Hatayspor logo"],
    "pendikspor": ["Pendikspor forma", "Pendikspor mac", "Pendikspor futbol", "Pendikspor logo"],
    "istanbulspor": ["Istanbulspor forma", "Istanbulspor mac", "Istanbulspor futbol", "Istanbulspor logo"],
    "ankaraguru": ["Ankaragucu forma", "MKE Ankaragucu mac", "Ankaragucu futbol", "Ankaragucu logo"],
    # ==============================
    # AVRUPA BUYUK TAKIMLAR
    # ==============================
    # ISPANYA
    "realmadrid": ["Real Madrid forma", "Real Madrid mac", "Real Madrid futbol", "Real Madrid logo"],
    "barcelona": ["FC Barcelona forma", "Barcelona mac", "Barcelona futbol", "Barcelona logo"],
    "atleticomadrid": ["Atletico Madrid forma", "Atletico Madrid mac", "Atletico Madrid futbol"],
    # INGILTERE
    "manchester_city": ["Manchester City forma", "Man City mac", "Manchester City futbol", "Man City logo"],
    "manchester_united": ["Manchester United forma", "Man United mac", "Manchester United futbol"],
    "liverpool": ["Liverpool FC forma", "Liverpool mac", "Liverpool futbol", "Liverpool logo"],
    "chelsea": ["Chelsea FC forma", "Chelsea mac", "Chelsea futbol", "Chelsea logo"],
    "arsenal": ["Arsenal FC forma", "Arsenal mac", "Arsenal futbol", "Arsenal logo"],
    "tottenham": ["Tottenham Hotspur forma", "Tottenham mac", "Spurs futbol", "Tottenham logo"],
    # ALMANYA
    "bayernmunich": ["Bayern Munich forma", "Bayern Munchen mac", "Bayern Munich futbol", "Bayern logo"],
    "dortmund": ["Borussia Dortmund forma", "Dortmund mac", "BVB futbol", "Dortmund logo"],
    # ITALYA
    "juventus": ["Juventus forma", "Juventus mac", "Juventus futbol", "Juventus logo"],
    "acmilan": ["AC Milan forma", "AC Milan mac", "Milan futbol", "AC Milan logo"],
    "intermilan": ["Inter Milan forma", "Inter Milan mac", "Inter futbol", "Inter Milano logo"],
    "napoli": ["SSC Napoli forma", "Napoli mac", "Napoli futbol", "Napoli logo"],
    # FRANSA
    "psg": ["Paris Saint Germain forma", "PSG mac", "PSG futbol", "PSG logo"],
    # PORTEKIZ
    "benfica": ["Benfica forma", "Benfica mac", "SL Benfica futbol", "Benfica logo"],
    "porto": ["FC Porto forma", "Porto mac", "Porto futbol", "Porto logo"],
    # HOLLANDA
    "ajax": ["Ajax Amsterdam forma", "Ajax mac", "Ajax futbol", "Ajax logo"],
    # ==============================
    # FUTBOLCULAR - PREMIER LEAGUE
    # ==============================
    "haaland": ["Erling Haaland futbol", "Haaland gol", "Haaland Manchester City", "Haaland Norway"],
    "salah": ["Mohamed Salah futbol", "Salah gol", "Salah Liverpool", "Salah Egypt"],
    "debruyne": ["Kevin De Bruyne futbol", "De Bruyne gol", "De Bruyne Manchester City"],
    "saka": ["Bukayo Saka futbol", "Saka gol", "Saka Arsenal", "Saka England"],
    "odegaard": ["Martin Odegaard futbol", "Odegaard gol", "Odegaard Arsenal", "Odegaard Norway"],
    "palmer": ["Cole Palmer futbol", "Cole Palmer gol", "Palmer Chelsea", "Palmer England"],
    "rice": ["Declan Rice futbol", "Declan Rice Arsenal", "Rice gol", "Rice England"],
    "bruno": ["Bruno Fernandes futbol", "Bruno Fernandes gol", "Bruno Manchester United"],
    "sonheungmin": ["Son Heung-min futbol", "Son Tottenham gol", "Son Heung min Spurs"],
    "foden": ["Phil Foden futbol", "Foden gol", "Foden Manchester City", "Foden England"],
    "rashford": ["Marcus Rashford futbol", "Rashford gol", "Rashford Manchester United"],
    "alexander_arnold": ["Trent Alexander Arnold futbol", "Alexander Arnold Liverpool", "Trent gol"],
    # ==============================
    # FUTBOLCULAR - LA LIGA
    # ==============================
    "messi": ["Lionel Messi futbol", "Messi gol", "Messi Inter Miami", "Messi Argentina"],
    "mbappe": ["Kylian Mbappe futbol", "Mbappe gol", "Mbappe Real Madrid", "Mbappe France"],
    "vinicius": ["Vinicius Jr futbol", "Vinicius gol", "Vinicius Real Madrid", "Vinicius Brazil"],
    "bellingham": ["Jude Bellingham futbol", "Bellingham gol", "Bellingham Real Madrid"],
    "lewandowski": ["Robert Lewandowski futbol", "Lewandowski gol", "Lewandowski Barcelona"],
    "pedri": ["Pedri futbol", "Pedri Barcelona gol", "Pedri Spain", "Pedri midfielder"],
    "gavi": ["Gavi futbol", "Gavi Barcelona", "Gavi Spain", "Gavi gol"],
    "yamal": ["Lamine Yamal futbol", "Lamine Yamal Barcelona", "Yamal gol", "Yamal Spain"],
    "rodrygo": ["Rodrygo futbol", "Rodrygo Real Madrid gol", "Rodrygo Brazil"],
    "modric": ["Luka Modric futbol", "Modric Real Madrid", "Modric gol", "Modric Croatia"],
    "kroos": ["Toni Kroos futbol", "Kroos Real Madrid", "Kroos gol", "Kroos Germany"],
    "ardaguler": ["Arda Guler futbol", "Arda Guler Real Madrid", "Arda Guler gol", "Arda Guler milli takim"],
    # ==============================
    # FUTBOLCULAR - BUNDESLIGA
    # ==============================
    "musiala": ["Jamal Musiala futbol", "Musiala Bayern Munich gol", "Musiala Germany"],
    "sane": ["Leroy Sane futbol", "Sane Bayern Munich gol", "Sane Germany"],
    "kane": ["Harry Kane futbol", "Kane Bayern Munich gol", "Kane England", "Harry Kane Tottenham"],
    "wirtz": ["Florian Wirtz futbol", "Wirtz Leverkusen gol", "Wirtz Germany"],
    "brandt": ["Julian Brandt futbol", "Brandt Dortmund", "Brandt gol", "Brandt Germany"],
    # ==============================
    # FUTBOLCULAR - SERIE A
    # ==============================
    "calhanoglu": ["Hakan Calhanoglu futbol", "Calhanoglu gol", "Calhanoglu Inter Milan"],
    "yildiz": ["Kenan Yildiz futbol", "Kenan Yildiz Juventus", "Kenan Yildiz gol"],
    "osimhen": ["Victor Osimhen futbol", "Osimhen gol", "Osimhen Napoli", "Osimhen Nigeria"],
    "lautaro": ["Lautaro Martinez futbol", "Lautaro Inter Milan gol", "Lautaro Argentina"],
    "leao": ["Rafael Leao futbol", "Leao AC Milan gol", "Leao Portugal"],
    "chiesa": ["Federico Chiesa futbol", "Chiesa gol", "Chiesa Juventus", "Chiesa Italy"],
    "vlahovic": ["Dusan Vlahovic futbol", "Vlahovic Juventus gol", "Vlahovic Serbia"],
    # ==============================
    # FUTBOLCULAR - LIGUE 1
    # ==============================
    "dembele": ["Ousmane Dembele futbol", "Dembele PSG gol", "Dembele France"],
    "hakimi": ["Achraf Hakimi futbol", "Hakimi PSG", "Hakimi gol", "Hakimi Morocco"],
    # ==============================
    # FUTBOLCULAR - TURKIYE
    # ==============================
    "icardi": ["Mauro Icardi futbol", "Icardi Galatasaray gol", "Icardi mac"],
    "dzeko": ["Edin Dzeko futbol", "Dzeko Fenerbahce gol", "Dzeko mac"],
    "talisca": ["Anderson Talisca futbol", "Talisca Fenerbahce gol", "Talisca mac"],
    "gedson": ["Gedson Fernandes futbol", "Gedson Besiktas", "Gedson gol"],
    "cenk_tosun": ["Cenk Tosun futbol", "Cenk Tosun Besiktas gol", "Cenk Tosun milli takim"],
    "irfancan": ["Irfan Can Kahveci futbol", "Irfan Can gol", "Irfan Can Fenerbahce"],
    "arda_turan": ["Arda Turan futbol", "Arda Turan Galatasaray", "Arda Turan Atletico Madrid"],
    "burak_yilmaz": ["Burak Yilmaz futbol", "Burak Yilmaz gol", "Burak Yilmaz milli takim"],
    "cengiz_under": ["Cengiz Under futbol", "Cengiz Under Fenerbahce gol", "Cengiz Under Roma"],
    "ugurcan": ["Ugurcan Cakir futbol", "Ugurcan kaleci", "Ugurcan Trabzonspor"],
    "hakansukur": ["Hakan Sukur futbol", "Hakan Sukur gol", "Hakan Sukur Galatasaray"],
    # ==============================
    # FUTBOLCULAR - EFSANELER
    # ==============================
    "ronaldo": ["Cristiano Ronaldo futbol", "Ronaldo gol", "Ronaldo Al Nassr", "Ronaldo Portugal"],
    "neymar": ["Neymar Jr futbol", "Neymar gol", "Neymar Brazil", "Neymar PSG"],
    "maradona": ["Diego Maradona futbol", "Maradona gol", "Maradona Argentina", "Maradona Napoli"],
    "pele": ["Pele futbol", "Pele gol", "Pele Brazil", "Pele legend footballer"],
    "zidane": ["Zinedine Zidane futbol", "Zidane gol", "Zidane Real Madrid", "Zidane France"],
    "ronaldinho": ["Ronaldinho futbol", "Ronaldinho gol", "Ronaldinho Barcelona", "Ronaldinho Brazil"],
    "beckham": ["David Beckham futbol", "Beckham gol", "Beckham Real Madrid", "Beckham Manchester United"],
    "henry": ["Thierry Henry futbol", "Henry gol", "Henry Arsenal", "Henry Barcelona"],
    "rooney": ["Wayne Rooney futbol", "Rooney gol", "Rooney Manchester United", "Rooney England"],
    "ibrahimovic": ["Zlatan Ibrahimovic futbol", "Ibrahimovic gol", "Ibrahimovic AC Milan"],
    # ==============================
    # SPOR TURLERI
    # ==============================
    "futbol": ["futbol mac", "futbol topu", "futbol sahasi", "soccer game"],
    "basketbol": ["basketbol mac", "basketball game", "NBA basketball", "basketbol oyuncusu"],
    "tenis": ["tenis mac", "tennis match", "tennis player", "tenis raketi"],
    "voleybol": ["voleybol mac", "volleyball game", "volleyball player", "voleybol takim"],
}


def download_images(class_name, search_queries, max_total):
    """Bir sinif icin gorselleri indir"""
    temp_dir = os.path.join(DATA_DIR, "temp", class_name)
    os.makedirs(temp_dir, exist_ok=True)

    per_query = max_total // len(search_queries) + 1

    for query in search_queries:
        print(f"  Araniyor: '{query}' ({per_query} gorsel)")
        crawler = BingImageCrawler(
            storage={"root_dir": temp_dir},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=per_query,
            min_size=(100, 100),
        )

    return temp_dir


def filter_images(source_dir):
    """Bozuk ve cok kucuk gorselleri filtrele"""
    valid = []
    removed = 0

    if not os.path.exists(source_dir):
        return valid

    for fname in os.listdir(source_dir):
        fpath = os.path.join(source_dir, fname)
        if not os.path.isfile(fpath):
            continue

        # Dosya boyutu kontrolu
        if os.path.getsize(fpath) < MIN_FILE_SIZE:
            os.remove(fpath)
            removed += 1
            continue

        # Gorsel acilabilirlik kontrolu
        try:
            img = Image.open(fpath)
            img.verify()
            # Tekrar ac (verify sonrasi kapaniyor)
            img = Image.open(fpath)
            img = img.convert("RGB")
            w, h = img.size
            if w < 50 or h < 50:
                os.remove(fpath)
                removed += 1
                continue
            valid.append(fpath)
        except Exception:
            os.remove(fpath)
            removed += 1

    if removed > 0:
        print(f"    {removed} bozuk/kucuk gorsel silindi")
    return valid


def split_train_val(valid_files, class_name):
    """Gorselleri train/val olarak ayir"""
    random.shuffle(valid_files)

    split_idx = int(len(valid_files) * TRAIN_RATIO)
    train_files = valid_files[:split_idx]
    val_files = valid_files[split_idx:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    val_class_dir = os.path.join(VAL_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for i, fpath in enumerate(train_files):
        ext = os.path.splitext(fpath)[1] or ".jpg"
        dst = os.path.join(train_class_dir, f"{class_name}_{i:04d}{ext}")
        shutil.copy2(fpath, dst)

    for i, fpath in enumerate(val_files):
        ext = os.path.splitext(fpath)[1] or ".jpg"
        dst = os.path.join(val_class_dir, f"{class_name}_val_{i:04d}{ext}")
        shutil.copy2(fpath, dst)

    return len(train_files), len(val_files)


def main():
    print("=" * 60)
    print("  OZEL MODEL - GORSEL TOPLAMA")
    print("=" * 60)

    # Temiz baslangic
    temp_dir = os.path.join(DATA_DIR, "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    total_train = 0
    total_val = 0

    for class_name, queries in CLASSES.items():
        print(f"\n[{class_name.upper()}] Gorseller indiriliyor...")
        temp_class_dir = download_images(class_name, queries, IMAGES_PER_CLASS)

        print(f"  Gorseller filtreleniyor...")
        valid = filter_images(temp_class_dir)
        print(f"  Gecerli gorsel: {len(valid)}")

        n_train, n_val = split_train_val(valid, class_name)
        total_train += n_train
        total_val += n_val
        print(f"  Train: {n_train} | Val: {n_val}")

    # Temp klasoru temizle
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print(f"  TOPLAM: {total_train} train + {total_val} val = {total_train + total_val} gorsel")
    print(f"  Klasor: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
