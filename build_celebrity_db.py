"""
Ünlü Yüz Veritabanı Oluşturucu
- Ünlülerin fotoğraflarını internetten indirir
- InsightFace ile yüz embedding'lerini çıkarır
- face_db.pkl dosyasına kaydeder
"""

import os
import pickle
import cv2
import numpy as np
from icrawler.builtin import BingImageCrawler
import insightface
from insightface.app import FaceAnalysis
import shutil
import time

# ===== AYARLAR =====
PHOTOS_PER_PERSON = 8        # Kişi başı indirilecek fotoğraf
MIN_EMBEDDINGS = 2           # Minimum geçerli embedding sayısı
FACE_DB_PATH = "face_db.pkl"
TEMP_DIR = "celebrity_temp_photos"

# ===== ÜNLÜ LİSTESİ =====
CELEBRITIES = {
    # === FUTBOLCULAR ===
    "Lionel Messi": "Lionel Messi face portrait",
    "Cristiano Ronaldo": "Cristiano Ronaldo face portrait",
    "Neymar Jr": "Neymar Jr face portrait",
    "Kylian Mbappe": "Kylian Mbappe face portrait",
    "Erling Haaland": "Erling Haaland face portrait",
    "Mohamed Salah": "Mohamed Salah footballer face",
    "Robert Lewandowski": "Robert Lewandowski face portrait",
    "Kevin De Bruyne": "Kevin De Bruyne face portrait",
    "Luka Modric": "Luka Modric face portrait",
    "Karim Benzema": "Karim Benzema face portrait",
    "Vinicius Jr": "Vinicius Junior Real Madrid face",
    "Jude Bellingham": "Jude Bellingham Real Madrid face",
    "Zlatan Ibrahimovic": "Zlatan Ibrahimovic face portrait",
    "David Beckham": "David Beckham face portrait",
    "Ronaldinho": "Ronaldinho footballer face portrait",
    "Zinedine Zidane": "Zinedine Zidane face portrait",
    "Ronaldo Nazario": "Ronaldo Nazario R9 face portrait",
    "Thierry Henry": "Thierry Henry face portrait",
    "Wayne Rooney": "Wayne Rooney face portrait",
    "Andres Iniesta": "Andres Iniesta face portrait",
    "Xavi Hernandez": "Xavi Hernandez face portrait",
    "Sergio Ramos": "Sergio Ramos face portrait",
    "Gerard Pique": "Gerard Pique face portrait",
    "Toni Kroos": "Toni Kroos face portrait",
    "Manuel Neuer": "Manuel Neuer face portrait",
    "Virgil van Dijk": "Virgil van Dijk face portrait",
    "Bruno Fernandes": "Bruno Fernandes Manchester United face",
    "Marcus Rashford": "Marcus Rashford face portrait",
    "Phil Foden": "Phil Foden face portrait",
    "Bukayo Saka": "Bukayo Saka Arsenal face",
    "Harry Kane": "Harry Kane face portrait",
    "Son Heung-min": "Son Heung-min Tottenham face",
    "Sadio Mane": "Sadio Mane face portrait",
    "Paul Pogba": "Paul Pogba face portrait",
    "Antoine Griezmann": "Antoine Griezmann face portrait",
    "Sergio Aguero": "Sergio Aguero face portrait",
    "Gareth Bale": "Gareth Bale face portrait",
    "Eden Hazard": "Eden Hazard face portrait",
    "Lamine Yamal": "Lamine Yamal Barcelona face",
    "Pedri": "Pedri Gonzalez Barcelona face",
    "Florian Wirtz": "Florian Wirtz Leverkusen face",
    "Jamal Musiala": "Jamal Musiala Bayern face",
    "Federico Valverde": "Federico Valverde Real Madrid face",
    "Rodri": "Rodri Hernandez Manchester City face",
    "Declan Rice": "Declan Rice Arsenal face",
    "Martin Odegaard": "Martin Odegaard Arsenal face",
    "Victor Osimhen": "Victor Osimhen face portrait",
    "Pele": "Pele footballer legend face",
    "Diego Maradona": "Diego Maradona face portrait",
    "Andrea Pirlo": "Andrea Pirlo face portrait",
    "Gianluigi Buffon": "Gianluigi Buffon face portrait",
    "Alessandro Del Piero": "Alessandro Del Piero face portrait",
    "Francesco Totti": "Francesco Totti face portrait",
    "Didier Drogba": "Didier Drogba face portrait",
    "Frank Lampard": "Frank Lampard face portrait",
    "Steven Gerrard": "Steven Gerrard face portrait",
    "Ryan Giggs": "Ryan Giggs face portrait",
    "Neymar Sr": "Neymar father face",
    "Lewandowski": "Robert Lewandowski Barcelona face close",
    "Thibaut Courtois": "Thibaut Courtois face portrait",
    "Alisson Becker": "Alisson Becker Liverpool face",
    "Trent Alexander-Arnold": "Trent Alexander-Arnold face portrait",
    "Joao Cancelo": "Joao Cancelo face portrait",
    "Bernardo Silva": "Bernardo Silva Manchester City face",
    "Joshua Kimmich": "Joshua Kimmich Bayern face",
    "Alphonso Davies": "Alphonso Davies Bayern face",
    "Achraf Hakimi": "Achraf Hakimi PSG face",
    "Ousmane Dembele": "Ousmane Dembele face portrait",
    "Raphinha": "Raphinha Barcelona face",
    "Darwin Nunez": "Darwin Nunez Liverpool face",
    "Alejandro Garnacho": "Alejandro Garnacho face portrait",
    "Cole Palmer": "Cole Palmer Chelsea face",
    "Kobbie Mainoo": "Kobbie Mainoo Manchester United face",

    # === TÜRK FUTBOLCULAR ===
    "Arda Guler": "Arda Guler Real Madrid face portrait",
    "Hakan Calhanoglu": "Hakan Calhanoglu Inter face portrait",
    "Cengiz Under": "Cengiz Under face portrait",
    "Irfan Can Kahveci": "Irfan Can Kahveci face portrait",
    "Hakan Sukur": "Hakan Sukur footballer face",
    "Rustu Recber": "Rustu Recber goalkeeper face",
    "Emre Belozoglu": "Emre Belozoglu face portrait",
    "Arda Turan": "Arda Turan face portrait",
    "Burak Yilmaz": "Burak Yilmaz footballer face",
    "Cenk Tosun": "Cenk Tosun face portrait",
    "Kerem Akturkoglu": "Kerem Akturkoglu Galatasaray face",
    "Ferdi Kadioglu": "Ferdi Kadioglu Brighton face",
    "Orkun Kokcu": "Orkun Kokcu Benfica face",
    "Mauro Icardi": "Mauro Icardi Galatasaray face",
    "Dries Mertens": "Dries Mertens Galatasaray face",
    "Edin Dzeko": "Edin Dzeko Fenerbahce face",
    "Dusan Tadic": "Dusan Tadic Fenerbahce face",

    # === TEKNİK DİREKTÖRLER ===
    "Pep Guardiola": "Pep Guardiola coach face portrait",
    "Carlo Ancelotti": "Carlo Ancelotti coach face portrait",
    "Jurgen Klopp": "Jurgen Klopp coach face portrait",
    "Jose Mourinho": "Jose Mourinho coach face portrait",
    "Sir Alex Ferguson": "Alex Ferguson coach face portrait",
    "Arsene Wenger": "Arsene Wenger coach face portrait",
    "Fatih Terim": "Fatih Terim coach face portrait",
    "Senol Gunes": "Senol Gunes coach face portrait",
    "Okan Buruk": "Okan Buruk coach face portrait",

    # === HOLLYWOOD AKTÖRLERİ ===
    "Leonardo DiCaprio": "Leonardo DiCaprio face portrait",
    "Brad Pitt": "Brad Pitt face portrait",
    "Tom Cruise": "Tom Cruise face portrait",
    "Johnny Depp": "Johnny Depp face portrait",
    "Robert Downey Jr": "Robert Downey Jr face portrait",
    "Chris Hemsworth": "Chris Hemsworth face portrait",
    "Chris Evans": "Chris Evans actor Captain America face",
    "Scarlett Johansson": "Scarlett Johansson face portrait",
    "Angelina Jolie": "Angelina Jolie face portrait",
    "Jennifer Aniston": "Jennifer Aniston face portrait",
    "Margot Robbie": "Margot Robbie face portrait",
    "Tom Hanks": "Tom Hanks face portrait",
    "Morgan Freeman": "Morgan Freeman face portrait",
    "Denzel Washington": "Denzel Washington face portrait",
    "Will Smith": "Will Smith actor face portrait",
    "Dwayne Johnson": "Dwayne Johnson The Rock face portrait",
    "Ryan Reynolds": "Ryan Reynolds face portrait",
    "Keanu Reeves": "Keanu Reeves face portrait",
    "Al Pacino": "Al Pacino face portrait",
    "Robert De Niro": "Robert De Niro face portrait",
    "Matt Damon": "Matt Damon face portrait",
    "Christian Bale": "Christian Bale face portrait",
    "Joaquin Phoenix": "Joaquin Phoenix face portrait",
    "Cillian Murphy": "Cillian Murphy Oppenheimer face",
    "Timothee Chalamet": "Timothee Chalamet face portrait",
    "Zendaya": "Zendaya actress face portrait",
    "Emma Stone": "Emma Stone actress face portrait",
    "Anne Hathaway": "Anne Hathaway actress face portrait",
    "Natalie Portman": "Natalie Portman face portrait",
    "Jennifer Lawrence": "Jennifer Lawrence face portrait",
    "Gal Gadot": "Gal Gadot face portrait",
    "Jason Momoa": "Jason Momoa face portrait",
    "Henry Cavill": "Henry Cavill face portrait",
    "Benedict Cumberbatch": "Benedict Cumberbatch face portrait",
    "Tom Holland": "Tom Holland Spider-Man actor face",
    "Florence Pugh": "Florence Pugh actress face portrait",
    "Ana de Armas": "Ana de Armas actress face portrait",
    "Samuel L Jackson": "Samuel L Jackson face portrait",
    "Harrison Ford": "Harrison Ford face portrait",
    "Meryl Streep": "Meryl Streep face portrait",
    "Cate Blanchett": "Cate Blanchett face portrait",
    "Sandra Bullock": "Sandra Bullock face portrait",
    "Julia Roberts": "Julia Roberts face portrait",
    "Nicole Kidman": "Nicole Kidman face portrait",
    "Emma Watson": "Emma Watson actress face portrait",
    "Daniel Craig": "Daniel Craig James Bond face",
    "Idris Elba": "Idris Elba face portrait",
    "Pedro Pascal": "Pedro Pascal actor face portrait",
    "Oscar Isaac": "Oscar Isaac actor face portrait",
    "Hugh Jackman": "Hugh Jackman Wolverine face portrait",
    "Ryan Gosling": "Ryan Gosling face portrait",
    "Chris Pratt": "Chris Pratt actor face portrait",
    "Jason Statham": "Jason Statham face portrait",
    "Vin Diesel": "Vin Diesel face portrait",
    "Sylvester Stallone": "Sylvester Stallone face portrait",
    "Arnold Schwarzenegger": "Arnold Schwarzenegger face portrait",
    "Jackie Chan": "Jackie Chan face portrait",
    "Jim Carrey": "Jim Carrey face portrait",
    "Adam Sandler": "Adam Sandler face portrait",
    "Robin Williams": "Robin Williams actor face portrait",
    "Mark Ruffalo": "Mark Ruffalo face portrait",
    "Paul Rudd": "Paul Rudd actor face portrait",
    "Charlize Theron": "Charlize Theron face portrait",
    "Halle Berry": "Halle Berry face portrait",
    "Viola Davis": "Viola Davis actress face portrait",
    "Austin Butler": "Austin Butler actor face portrait",
    "Jeremy Renner": "Jeremy Renner face portrait",
    "Salma Hayek": "Salma Hayek face portrait",
    "Penelope Cruz": "Penelope Cruz face portrait",
    "Javier Bardem": "Javier Bardem face portrait",
    "Antonio Banderas": "Antonio Banderas face portrait",
    "Mads Mikkelsen": "Mads Mikkelsen face portrait",
    "Keanu Reeves": "Keanu Reeves John Wick face",

    # === TÜRK ÜNLÜLER ===
    "Kemal Sunal": "Kemal Sunal actor face portrait",
    "Cem Yilmaz": "Cem Yilmaz comedian face portrait",
    "Ata Demirer": "Ata Demirer face portrait",
    "Sahan Gokbakar": "Sahan Gokbakar face portrait",
    "Kivanc Tatlitug": "Kivanc Tatlitug face portrait",
    "Burak Ozcivit": "Burak Ozcivit face portrait",
    "Can Yaman": "Can Yaman actor face portrait",
    "Cagatay Ulusoy": "Cagatay Ulusoy face portrait",
    "Beren Saat": "Beren Saat actress face portrait",
    "Fahriye Evcen": "Fahriye Evcen face portrait",
    "Hande Ercel": "Hande Ercel face portrait",
    "Berguzar Korel": "Berguzar Korel face portrait",
    "Turkan Soray": "Turkan Soray actress face portrait",
    "Halit Ergenc": "Halit Ergenc actor face portrait",
    "Engin Akyurek": "Engin Akyurek face portrait",
    "Ibrahim Celikkol": "Ibrahim Celikkol face portrait",
    "Demet Ozdemir": "Demet Ozdemir face portrait",
    "Elcin Sangu": "Elcin Sangu face portrait",
    "Hazal Kaya": "Hazal Kaya actress face portrait",
    "Neslihan Atagul": "Neslihan Atagul face portrait",
    "Acun Ilicali": "Acun Ilicali face portrait",
    "Sener Sahin": "Sener Sahin turkish actor face",
    "Adile Nasit": "Adile Nasit actress face",
    "Muammer Ozer": "Muammer Ozer actor face",
    "Tarik Akan": "Tarik Akan actor face portrait",
    "Cuneyt Arkin": "Cuneyt Arkin actor face portrait",
    "Turkan Soray": "Turkan Soray face close",

    # === TÜRK MÜZİSYENLER ===
    "Tarkan": "Tarkan turkish singer face portrait",
    "Sezen Aksu": "Sezen Aksu singer face portrait",
    "Ibrahim Tatlises": "Ibrahim Tatlises singer face portrait",
    "Ajda Pekkan": "Ajda Pekkan singer face portrait",
    "Hadise": "Hadise singer face portrait",
    "Gulsen": "Gulsen turkish singer face portrait",
    "Murat Boz": "Murat Boz singer face portrait",
    "Aleyna Tilki": "Aleyna Tilki singer face portrait",
    "Ebru Gundes": "Ebru Gundes singer face portrait",
    "Serdar Ortac": "Serdar Ortac singer face portrait",
    "Mustafa Sandal": "Mustafa Sandal singer face portrait",
    "Simge Sagin": "Simge Sagin singer face portrait",
    "Muazzez Ersoy": "Muazzez Ersoy singer face portrait",
    "Kenan Dogulu": "Kenan Dogulu singer face portrait",
    "Manga": "Manga turkish band Ferman face",
    "Teoman": "Teoman turkish singer face portrait",
    "Sertab Erener": "Sertab Erener singer face portrait",
    "Nil Karaibrahimgil": "Nil Karaibrahimgil singer face",
    "Demet Akalin": "Demet Akalin singer face portrait",
    "Hande Yener": "Hande Yener singer face portrait",
    "Bengü": "Bengu turkish singer face portrait",
    "Mabel Matiz": "Mabel Matiz singer face portrait",
    "Mor ve Otesi Harun": "Harun Tekin Mor ve Otesi face",

    # === MÜZİSYENLER (DÜNYA) ===
    "Taylor Swift": "Taylor Swift face portrait",
    "Beyonce": "Beyonce face portrait",
    "Rihanna": "Rihanna face portrait",
    "Drake": "Drake rapper face portrait",
    "Eminem": "Eminem rapper face portrait",
    "Kanye West": "Kanye West face portrait",
    "Ed Sheeran": "Ed Sheeran face portrait",
    "Adele": "Adele singer face portrait",
    "Billie Eilish": "Billie Eilish face portrait",
    "Ariana Grande": "Ariana Grande face portrait",
    "Justin Bieber": "Justin Bieber face portrait",
    "Selena Gomez": "Selena Gomez face portrait",
    "Dua Lipa": "Dua Lipa face portrait",
    "Harry Styles": "Harry Styles face portrait",
    "The Weeknd": "The Weeknd singer Abel face portrait",
    "Bruno Mars": "Bruno Mars face portrait",
    "Lady Gaga": "Lady Gaga face portrait",
    "Shakira": "Shakira face portrait",
    "Madonna": "Madonna singer face portrait",
    "Post Malone": "Post Malone face portrait",
    "Travis Scott": "Travis Scott rapper face portrait",
    "Doja Cat": "Doja Cat face portrait",
    "Bad Bunny": "Bad Bunny singer face portrait",
    "Olivia Rodrigo": "Olivia Rodrigo face portrait",
    "Miley Cyrus": "Miley Cyrus face portrait",
    "Katy Perry": "Katy Perry face portrait",
    "Nicki Minaj": "Nicki Minaj face portrait",
    "Cardi B": "Cardi B face portrait",
    "Kendrick Lamar": "Kendrick Lamar face portrait",
    "50 Cent": "50 Cent rapper face portrait",
    "Jay-Z": "Jay-Z rapper face portrait",
    "Lana Del Rey": "Lana Del Rey face portrait",
    "SZA": "SZA singer face portrait",
    "Ice Spice": "Ice Spice rapper face portrait",
    "Sabrina Carpenter": "Sabrina Carpenter face portrait",
    "Chappell Roan": "Chappell Roan singer face portrait",

    # === BASKETBOLCULAR ===
    "LeBron James": "LeBron James face portrait",
    "Stephen Curry": "Stephen Curry face portrait",
    "Kevin Durant": "Kevin Durant face portrait",
    "Giannis Antetokounmpo": "Giannis Antetokounmpo face portrait",
    "Luka Doncic": "Luka Doncic face portrait",
    "Nikola Jokic": "Nikola Jokic face portrait",
    "Michael Jordan": "Michael Jordan basketball face portrait",
    "Kobe Bryant": "Kobe Bryant face portrait",
    "Shaquille O'Neal": "Shaquille O'Neal face portrait",

    # === LİDERLER / İŞ İNSANLARI ===
    "Mustafa Kemal Ataturk": "Mustafa Kemal Ataturk face portrait",
    "Recep Tayyip Erdogan": "Recep Tayyip Erdogan face portrait",
    "Donald Trump": "Donald Trump face portrait",
    "Barack Obama": "Barack Obama face portrait",
    "Joe Biden": "Joe Biden face portrait",
    "Vladimir Putin": "Vladimir Putin face portrait",
    "Emmanuel Macron": "Emmanuel Macron face portrait",
    "Angela Merkel": "Angela Merkel face portrait",
    "Elon Musk": "Elon Musk face portrait",
    "Jeff Bezos": "Jeff Bezos face portrait",
    "Bill Gates": "Bill Gates face portrait",
    "Mark Zuckerberg": "Mark Zuckerberg face portrait",
    "Steve Jobs": "Steve Jobs Apple face portrait",
    "Albert Einstein": "Albert Einstein face portrait",
    "Nelson Mandela": "Nelson Mandela face portrait",
    "Pope Francis": "Pope Francis face portrait",
    "Queen Elizabeth II": "Queen Elizabeth II face portrait",
    "King Charles III": "King Charles III face portrait",
    "Princess Diana": "Princess Diana face portrait",

    # === DİĞER SPORCULAR ===
    "Roger Federer": "Roger Federer face portrait",
    "Rafael Nadal": "Rafael Nadal face portrait",
    "Novak Djokovic": "Novak Djokovic face portrait",
    "Serena Williams": "Serena Williams face portrait",
    "Usain Bolt": "Usain Bolt face portrait",
    "Mike Tyson": "Mike Tyson face portrait",
    "Floyd Mayweather": "Floyd Mayweather face portrait",
    "Conor McGregor": "Conor McGregor face portrait",
    "Lewis Hamilton": "Lewis Hamilton F1 face portrait",
    "Max Verstappen": "Max Verstappen F1 face portrait",
    "Tiger Woods": "Tiger Woods golf face portrait",
    "Michael Phelps": "Michael Phelps swimmer face portrait",
    "Simone Biles": "Simone Biles gymnast face portrait",
    "Naomi Osaka": "Naomi Osaka tennis face portrait",
}


def download_images(name, query, output_dir, max_num=PHOTOS_PER_PERSON):
    """Bing'den fotoğraf indir"""
    person_dir = os.path.join(output_dir, name.replace(" ", "_"))
    if os.path.exists(person_dir):
        shutil.rmtree(person_dir)
    os.makedirs(person_dir, exist_ok=True)

    try:
        crawler = BingImageCrawler(
            storage={"root_dir": person_dir},
            log_level=50  # Sadece kritik hatalar
        )
        crawler.crawl(
            keyword=query,
            max_num=max_num,
            min_size=(100, 100),
        )
    except Exception as e:
        print(f"  [HATA] {name}: {e}")

    return person_dir


def extract_embeddings(face_app, person_dir):
    """Bir kişinin fotoğraflarından yüz embedding'leri çıkar"""
    embeddings = []

    if not os.path.exists(person_dir):
        return embeddings

    for fname in sorted(os.listdir(person_dir)):
        fpath = os.path.join(person_dir, fname)
        try:
            img = cv2.imread(fpath)
            if img is None:
                continue

            faces = face_app.get(img)
            if len(faces) == 0:
                continue

            # En büyük yüzü al
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.normed_embedding

            if emb is not None:
                embeddings.append(emb)

        except Exception:
            continue

    return embeddings


def main():
    print("=" * 60)
    print("  ÜNLÜ YÜZ VERİTABANI OLUŞTURUCU")
    print(f"  {len(CELEBRITIES)} kişi işlenecek")
    print("=" * 60)

    # Mevcut veritabanını yükle
    face_db = {}
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "rb") as f:
            face_db = pickle.load(f)
        print(f"\nMevcut veritabanı yüklendi: {len(face_db)} kişi")

    # InsightFace başlat
    print("\nInsightFace yükleniyor...")
    face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace hazır!\n")

    # Geçici klasör
    os.makedirs(TEMP_DIR, exist_ok=True)

    success = 0
    fail = 0
    skip = 0

    for i, (name, query) in enumerate(CELEBRITIES.items(), 1):
        # Zaten veritabanında varsa atla
        name_key = name.lower().replace(" ", "_")
        if name_key in face_db and len(face_db[name_key].get("embeddings", [])) >= MIN_EMBEDDINGS:
            print(f"[{i}/{len(CELEBRITIES)}] {name} - ZATEN MEVCUT ({len(face_db[name_key]['embeddings'])} embedding), atlanıyor")
            skip += 1
            continue

        print(f"[{i}/{len(CELEBRITIES)}] {name}...")
        print(f"  Fotoğraflar indiriliyor...", end=" ", flush=True)

        person_dir = download_images(name, query, TEMP_DIR)
        n_photos = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))]) if os.path.exists(person_dir) else 0
        print(f"{n_photos} fotoğraf indirildi.", end=" ", flush=True)

        if n_photos == 0:
            print("BAŞARISIZ - fotoğraf indirilemedi")
            fail += 1
            continue

        embeddings = extract_embeddings(face_app, person_dir)
        print(f"{len(embeddings)} yüz embedding çıkarıldı.", end=" ")

        if len(embeddings) >= MIN_EMBEDDINGS:
            face_db[name_key] = {
                "name": name,
                "embeddings": embeddings
            }
            print("OK ✓")
            success += 1
        else:
            print("BAŞARISIZ - yeterli yüz bulunamadı")
            fail += 1

        # Her 10 kişide bir kaydet
        if i % 10 == 0:
            with open(FACE_DB_PATH, "wb") as f:
                pickle.dump(face_db, f)
            print(f"  [KAYIT] Ara kayıt yapıldı ({len(face_db)} kişi)")

        # Rate limiting
        time.sleep(0.5)

    # Geçici dosyaları temizle
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    # Son kayıt
    with open(FACE_DB_PATH, "wb") as f:
        pickle.dump(face_db, f)

    print("\n" + "=" * 60)
    print(f"  TAMAMLANDI!")
    print(f"  Başarılı: {success}")
    print(f"  Başarısız: {fail}")
    print(f"  Atlandı (mevcut): {skip}")
    print(f"  Toplam veritabanı: {len(face_db)} kişi")
    print(f"  Dosya: {FACE_DB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
