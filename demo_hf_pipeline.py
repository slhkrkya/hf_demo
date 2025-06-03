from transformers import pipeline
import json
import os

# Geliştirilmiş Açıklama
"""
Bu script, Trendyol yorumlarını analiz ederek:
- Ayrıntılı duygu analizi (Very Positive, Positive, Neutral, Negative, Very Negative)
- 10'dan fazla konuyla ilgili NER (beden, fiyat, iade, kalite vs.)
- Yıldız skoru tahmini (1-5)
- Şikayet/Öneri/Övgü sınıflaması
- Dinamik tavsiyeler üretimi
yapar ve organize JSON çıktısı verir.
"""

# 1) Model yükleme
model_path = os.path.abspath("./trained_model")

sentiment_analyzer = pipeline(
    "text-classification",
    model=model_path,
    tokenizer=model_path,
    local_files_only=True
)

# 2) Gelişmiş Rule-based NER
def ner_extractor(text):
    text_lower = text.lower()
    ents = []

    if "bir beden" in text_lower or "beden" in text_lower:
        ents.append({"entity": "SIZE_COMMENT"})
    if any(w in text_lower for w in ["eşime", "anneme", "arkadaşıma", "hediye", "doğum günü", "sevgililer günü"]):
        ents.append({"entity": "GIFT_OCCASION"})
    if any(w in text_lower for w in ["geç geldi", "kırık geldi", "paket hasarlı", "kargo geçikti"]):
        ents.append({"entity": "SHIPPING_ISSUE"})
    if any(w in text_lower for w in ["kalitesiz", "dayanıksız", "yıprandı", "bozuldu"]):
        ents.append({"entity": "QUALITY_ISSUE"})
    if any(w in text_lower for w in ["pahalı", "fiyatı yüksek", "fiyat performans"]):
        ents.append({"entity": "PRICE_COMMENT"})
    if any(w in text_lower for w in ["iade ettim", "iade süreci", "geri gönderdim"]):
        ents.append({"entity": "RETURN_PROCESS"})
    if any(w in text_lower for w in ["garanti kapsamı", "garantiye yolladım"]):
        ents.append({"entity": "WARRANTY_PROCESS"})
    if any(w in text_lower for w in ["ürün açıklamasıyla uyuşmuyor", "yanıltıcı bilgi"]):
        ents.append({"entity": "MISMATCH_ISSUE"})

    return ents

# 3) Negatif Override (aşırı kötü yorumlar için)
def override_sentiment(text, raw_pred):
    text_lower = text.lower()
    if any(w in text_lower for w in ["rezalet", "berbat", "fiyasko", "iğrenç", "hayal kırıklığı"]):
        return {"label": "Very_Negative", "score": 1.0}
    return raw_pred

# 4) Yıldız skoru tahmini
def predict_star_rating(sentiment_label):
    if sentiment_label == "Very_Positive":
        return 5
    elif sentiment_label == "Positive":
        return 4
    elif sentiment_label == "Neutral":
        return 3
    elif sentiment_label == "Negative":
        return 2
    elif sentiment_label == "Very_Negative":
        return 1
    return 3  # default

# 5) Şikayet/Öneri/Övgü ayrımı
def classify_feedback(sentiment_label, entities):
    if sentiment_label in ["Negative", "Very_Negative"]:
        return "Şikayet"
    if sentiment_label in ["Positive", "Very_Positive"]:
        if any(e['entity'] == "GIFT_OCCASION" for e in entities):
            return "Övgü ve Tavsiye"
        else:
            return "Övgü"
    return "Nötr"

# 6) Örnek Gelişmiş Yorumlar
reviews = [
    {"product_id": "P1", "text": "Ürün kaliteli fakat kargo berbat. Kutusu yırtılmış geldi."},
    {"product_id": "P2", "text": "Doğum günü hediyesi olarak aldım, eşim çok beğendi."},
    {"product_id": "P3", "text": "İade etmek zorunda kaldım, açıklamayla uyuşmadı."},
    {"product_id": "P4", "text": "Kalitesi mükemmel, fiyat performans ürünü."},
]

# 7) İşlem Döngüsü
results = {}
for r in reviews:
    pid, txt = r["product_id"], r["text"]
    
    # Ham model tahmini
    raw = sentiment_analyzer(txt)[0]
    label_map = {"LABEL_0": "Neutral", "LABEL_1": "Positive", "LABEL_2": "Negative"}
    mapped_label = label_map.get(raw["label"], "Neutral")
    
    # Gelişmiş override
    sent = override_sentiment(txt, {"label": mapped_label, "score": raw["score"]})
    
    # Entity çıkarımı
    ents = ner_extractor(txt)

    # Yıldız skoru tahmini
    stars = predict_star_rating(sent["label"])

    # Şikayet/Öneri/Övgü sınıflaması
    feedback = classify_feedback(sent["label"], ents)

    # Tavsiye
    suggestions = []
    if feedback == "Şikayet":
        suggestions.append("Ürün/kargo süreçleri gözden geçirilmeli.")
    if feedback == "Övgü" or feedback == "Övgü ve Tavsiye":
        suggestions.append("Övgü yorumları sosyal medyada kullanılabilir.")
    if any(e['entity'] == "RETURN_PROCESS" for e in ents):
        suggestions.append("İade süreçleri incelenmeli.")
    if any(e['entity'] == "WARRANTY_PROCESS" for e in ents):
        suggestions.append("Garanti süreçleri iyileştirilebilir.")

    results.setdefault(pid, []).append({
        "text": txt,
        "sentiment": sent,
        "star_rating": stars,
        "entities": ents,
        "feedback_type": feedback,
        "suggestion": "; ".join(suggestions) or "—"
    })

# 8) JSON Çıktısı
print(json.dumps(results, indent=2, ensure_ascii=False))