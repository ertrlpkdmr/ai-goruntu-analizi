"""
Eğittiğimiz NLP duygu analizi modelini kullan.
Kullanım:
  python predict_nlp.py "Takim bugun harika oynadi"
  python predict_nlp.py            # interaktif mod
"""
import sys
import pickle

with open("nlp_sentiment_model.pkl", "rb") as f:
    d = pickle.load(f)
model, ad = d["model"], d["model_adi"]
print(f"Model: {ad} (TF-IDF + klasik ML)\n")

def tahmin(metin):
    p = model.predict([metin])[0]
    prob = model.predict_proba([metin])[0].max()
    etiket = "OLUMLU 😊" if p == 1 else "OLUMSUZ 😞"
    print(f"  [{etiket}]  güven %{prob*100:.0f}  ->  \"{metin}\"")

if len(sys.argv) > 1:
    tahmin(" ".join(sys.argv[1:]))
else:
    print("Bir yorum yaz (cikmak icin bos birak):")
    while True:
        try:
            t = input("> ").strip()
        except EOFError:
            break
        if not t:
            break
        tahmin(t)
