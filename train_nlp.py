"""
Türkçe Spor Yorumu Duygu Analizi — KENDİ EĞİTTİĞİMİZ NLP MODELİ
NLP yöntemi: TF-IDF (metni sayıya çevirme) + Klasik ML sınıflandırıcı
Karşılaştırma: Lojistik Regresyon vs Random Forest vs Naive Bayes

Kullanım: python train_nlp.py
"""
import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# ---- 1. VERİ ----
with open("nlp_data.json", encoding="utf-8") as f:
    data = json.load(f)
ornekler = data["ornekler"]
metinler = [x[0] for x in ornekler]
etiketler = np.array([x[1] for x in ornekler])
print(f"Veri seti: {len(metinler)} yorum  "
      f"({int(etiketler.sum())} olumlu, {int((etiketler==0).sum())} olumsuz)\n")

# ---- 2. TF-IDF (NLP özellik çıkarımı) ----
# Metni sayısal vektöre çevirir: her kelime/ikili bir özellik olur.
# 1-2 gram: tekil kelimeler + ikili kelime grupları ("hayal kirikligi" gibi)
vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                             min_df=1, max_df=0.9)

# ---- 3. Eğitim/test ayrımı ----
X_tr, X_te, y_tr, y_te = train_test_split(
    metinler, etiketler, test_size=0.25, random_state=42, stratify=etiketler)
print(f"Eğitim: {len(X_tr)} | Test: {len(X_te)}\n")

# ---- 4. ÜÇ KLASİK ML ALGORİTMASINI KARŞILAŞTIR ----
modeller = {
    "Lojistik Regresyon": LogisticRegression(max_iter=1000, C=3.0),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=0.3),
}

print("=" * 58)
print(f"{'Model':<22}{'Test Doğr.':<14}{'5-kat CV (ort)':<16}")
print("=" * 58)

sonuclar = {}
for ad, clf in modeller.items():
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])
    # Çapraz doğrulama (küçük veri için güvenilir ölçüm)
    cv = cross_val_score(pipe, metinler, etiketler, cv=5)
    # Eğitim/test
    pipe.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, pipe.predict(X_te))
    sonuclar[ad] = (acc, cv.mean(), pipe)
    print(f"{ad:<22}%{acc*100:<12.1f}%{cv.mean()*100:.1f} (±{cv.std()*100:.1f})")

print("=" * 58)

# ---- 5. EN İYİ MODELİ SEÇ + DETAYLI RAPOR ----
en_iyi_ad = max(sonuclar, key=lambda k: sonuclar[k][1])  # CV'ye göre
en_iyi = sonuclar[en_iyi_ad][2]
print(f"\nEN İYİ MODEL: {en_iyi_ad} (CV %{sonuclar[en_iyi_ad][1]*100:.1f})\n")
print("Test seti detaylı raporu:")
print(classification_report(y_te, en_iyi.predict(X_te),
                            target_names=["olumsuz", "olumlu"], digits=3))

# ---- 6. KAYDET ----
with open("nlp_sentiment_model.pkl", "wb") as f:
    pickle.dump({"model": en_iyi, "model_adi": en_iyi_ad}, f)
print("Model kaydedildi: nlp_sentiment_model.pkl")

# ---- 7. CANLI ÖRNEK ----
print("\n--- Eğitilen modelle örnek tahminler ---")
testler = [
    "Takim bugun harika oynadi gercekten mukemmeldi",
    "Berbat bir maçti tam bir hayal kirikligi",
    "Kaleci muhtesem kurtarislar yapti",
    "Hakem rezalet kararlar verdi maçi mahvetti",
]
for t in testler:
    p = en_iyi.predict([t])[0]
    prob = en_iyi.predict_proba([t])[0].max()
    print(f"  [{'OLUMLU' if p==1 else 'OLUMSUZ'}] (%{prob*100:.0f})  \"{t}\"")
