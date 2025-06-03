import os
import re
import json
import torch, asyncio
from functools import partial
from typing import List, Dict

import httpx
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import DuplicateKeyError
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv

# İnsan okunur duygu etiketleri
SENTIMENT_LABELS = {
    "LABEL_0": "Olumsuz",
    "LABEL_1": "Olumlu",
    "LABEL_2": "Nötr"
}

# Basit olumsuz kelime listesi
NEGATIVE_LEXICON = ["kötü","ağır","berbat","pişman","vasat","yazık","beklemeyin"]
def lexicon_negative(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in NEGATIVE_LEXICON)

# ----------------------------
#  JSON-LD yorum çıkarıcı
# ----------------------------
def extract_ldjson_reviews(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string)
        except Exception:
            continue
        if isinstance(data, dict) and data.get("@type") == "Product" and "review" in data:
            return data["review"]
        if isinstance(data, list) and all(item.get("@type") == "Review" for item in data):
            return data
    return None

# ----------------------------
#  Fallback: Window JSON
# ----------------------------
def extract_reviews_json(html: str):
    m = re.search(
        r"window\.__REVIEW_APP_INITIAL_STATE__\s*=\s*({.*?})\s*;\s*</script>",
        html,
        re.S
    )
    if not m:
        raise ValueError("Yorum JSON'u bulunamadı")
    return json.loads(m.group(1).strip())

# ----------------------------
#  Pydantic modeller
# ----------------------------
class Analysis(BaseModel):
    text: str
    sentiment: Dict
    entities: List[Dict]
    suggestion: str

class AnalysisResponse(BaseModel):
    product_id: str
    total: int
    limit: int
    offset: int
    analyses: List[Analysis]
    summary: str

# ----------------------------
#  Ortam & DB ayarları
# ----------------------------
load_dotenv()
MONGODB_URI     = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME         = os.getenv("MONGODB_DB", "hf_demo")
COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "reviews")

client      = AsyncIOMotorClient(MONGODB_URI)
db          = client[DB_NAME]
reviews_col = db[COLLECTION_NAME]

# ----------------------------
#  FastAPI & CORS
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def create_index():
    await reviews_col.delete_many({"text": None})
    try:
        await reviews_col.create_index(
            [("product_id", 1), ("text", 1)],
            unique=True
        )
    except Exception as e:
        print("Index oluşturulurken hata oldu (görmezden geliniyor):", e)

# ----------------------------
#  Duygu Analizi & NER
# ----------------------------
# 1) Model & Tokenizer
model_id = "anilguven/albert_tr_turkish_product_reviews"
model = AutoModelForSequenceClassification.from_pretrained(model_id)
# Dinamik quantization
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 2) Pipeline
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=-1,           # CPU’da çalıştır
    truncation=True,
    max_length=128
)

# 3) Async wrapper
async def analyze_sentiments(texts: list[str]) -> list[dict]:
    loop = asyncio.get_running_loop()
    fn = partial(sentiment_analyzer, texts, batch_size=16)
    return await loop.run_in_executor(None, fn)

def ner_extractor(text: str) -> List[Dict]:
    t = text.lower()
    ents: List[Dict] = []
    if "bir beden" in t:
        ents.append({"word": "bir beden", "entity": "SIZE_COMMENT"})
    if "sevgililer günü" in t:
        ents.append({"word": "sevgililer günü", "entity": "OCCASION"})
    if "eşime" in t:
        ents.append({"word": "eşime", "entity": "RECIPIENT"})
    return ents

# ----------------------------
#  Yardımcı fonksiyonlar
# ----------------------------
def extract_product_id(url: str) -> str:
    m = re.search(r"-p-(\d+)", url)
    if not m:
        raise ValueError("URL içinde '-p-<productId>' bulunamadı")
    return m.group(1)

# --------- Yardımcı: comment + model çıktısına göre etiket belirleme ------------
def classify_sentiment(text: str, raw: Dict) -> str:
    """
    Yorum metni ve model çıktısına göre duygu sınıflandırması yapar.
    """
    score = raw["score"]
    label = raw["label"]
    text_lower = text.lower()

    # Geliştirilmiş kalıplar
    strong_negatives = [
        "memnun kalmadım", "bir işe yaramıyor", "yararı yok",
        "hiç etkisi olmadı", "boşa para", "pişman oldum",
        "aldığıma pişmanım", "dökülmüştü", "hiçbir fayda sağlamadı",
        "hayal kırıklığına uğradım", "beklentimin çok altında",
        "çöp çıktı", "tam bir fiyasko", "kırık geldi", "eksik geldi",
        "hiç memnun değilim", "berbat", "çok kötü", "felaket",
        "işe yaramaz", "zarar verdi", "çok geç geldi", "yanlış ürün gönderildi",
        "kusurlu ürün", "defolu ürün", "kullanılamaz durumda",
        "kırık paket", "eksik parça", "iade ettim", "geri gönderdim",
        "sahte ürün", "orijinal değil", "aldatıcı açıklama",
        "hiç bir işe yaramıyor", "tam bir hayal kırıklığı", "rezalet",
        "dayanıksız", "zaten bozuk geldi", "2 kullanımda bozuldu",
        "bir hafta dayanmadı", "aldığım gibi çöpe attım"
    ]

    strong_positives = [
        "çok memnun kaldım", "harika ürün", "şiddetle tavsiye ederim",
        "kesinlikle alın", "beğendim", "çok güzel", "muhteşem",
        "beklentimin üstünde", "kusursuz", "harikulade", "bayıldım",
        "çok hızlı teslimat", "tam istediğim gibi", "beş yıldızlık ürün",
        "harika kalite", "çok iyi paketlenmiş", "gözüm kapalı alırım",
        "süper ürün", "inanılmaz kaliteli", "şahane", "efsane ürün",
        "parasını sonuna kadar hak ediyor", "kalitesi mükemmel",
        "hızlı kargo", "düşündüğümden daha iyi", "mükemmel paketleme",
        "çok işime yaradı", "çok fonksiyonel", "gerçekten tavsiye ederim",
        "kaliteli malzeme", "kullanımı çok rahat", "çok pratik",
        "düşünmeden alın", "piyasadaki en iyisi", "hayran kaldım",
        "kesinlikle tekrar alırım", "fiyatına göre harika",
        "kalitesinden ödün vermemiş"
    ]

    # 1) Lexicon temelli negatif kelimeler
    for kw in NEGATIVE_LEXICON:
        if re.search(rf"\b{kw}\b", text_lower):
            negation = re.search(rf"\b{kw}\s+(yapmıyor|değil|değildir)\b", text_lower)
            if not negation:
                return "Olumsuz"

    # 2) Güçlü kalıp sayımı
    neg_count = sum(1 for phrase in strong_negatives if phrase in text_lower)
    pos_count = sum(1 for phrase in strong_positives if phrase in text_lower)

    if neg_count >= 2:
        return "Olumsuz"
    if pos_count >= 2:
        return "Olumlu"

    # 3) Tek güçlü negatif/pozitif varsa
    if neg_count == 1:
        return "Olumsuz"
    if pos_count == 1:
        return "Olumlu"

    # 4) Model etiketi ve skora göre karar
    if label == "LABEL_1" and score > 0.8:
        return "Olumlu"
    elif label == "LABEL_0" and score > 0.8:
        return "Olumsuz"

    # 5) Düşük skorsa nötr
    if score < 0.6:
        return "Nötr"

    # 6) Kalanlar için model etiketi
    return SENTIMENT_LABELS.get(label, label)

# ----------------------------
#  Sağlık kontrol
# ----------------------------
@app.get("/ping")
async def ping():
    return {"ping": "pong"}

# ----------------------------
#  Yorumları çek & kaydet
# ----------------------------
@app.post("/fetch_by_url")
async def fetch_by_url(
    url: str = Query(..., description="Trendyol ürün detay sayfası veya internal API yorumlar URL’i")
):
    m_api = re.match(
        r"https?://apigw\.trendyol\.com/.+?/reviews/([^/]+)/([^/]+)-p-(\d+)/yorumlar\?(.+)",
        url
    )
    if m_api:
        brand, slug, pid, qs = m_api.groups()
        url = f"https://www.trendyol.com/{brand}/{slug}-p-{pid}?{qs}"

    m = re.match(r"https?://www\.trendyol\.com/([^/]+)/([^/]+)-p-(\d+)", url)
    if not m:
        raise HTTPException(400, "URL formatı hatalı.")
    brand, slug, pid = m.groups()

    qp = httpx.URL(url).params
    boutique, merchant = qp.get("boutiqueId"), qp.get("merchantId")
    if not (boutique and merchant):
        raise HTTPException(400, "URL içinde boutiqueId ve merchantId zorunlu")

    api_url = (
        f"https://apigw.trendyol.com/"
        f"discovery-web-socialgw-service/reviews/"
        f"{brand}/{slug}-p-{pid}/yorumlar"
    )
    base_params = {
        "boutiqueId": boutique, "merchantId": merchant,
        "sav": "true", "culture": "tr-TR",
        "storefrontId": 1, "RRsocialproofAbTesting": "A",
        "logged-in": "false", "isBuyer": "false",
        "channelId": 1
    }

    fetched = stored = 0

    async with httpx.AsyncClient(timeout=10.0) as client:
        p0 = {**base_params, "page": 0, "size": 30}
        r0 = await client.get(api_url, params=p0); r0.raise_for_status()
        j0 = r0.json()
        hs0 = j0["result"]["hydrateScript"]
        mo = re.search(r"window\.__REVIEW_APP_INITIAL_STATE__\s*=\s*({.*?});", hs0, re.S)
        if not mo:
            raise HTTPException(500, "HydrateScript içinde JSON bulunamadı")
        state0 = json.loads(mo.group(1))

        if isinstance(state0, list):
            reviews = state0
            fetched = len(reviews)
            for r in reviews:
                text = r.get("comment") or r.get("reviewBody","")
                if not text: continue
                doc = {
                    "product_id": pid,
                    "user":       r.get("userFullName") or r.get("author",{}).get("name",""),
                    "rating":     r.get("rate") or r.get("reviewRating",{}).get("ratingValue"),
                    "date":       r.get("commentDateISOtype") or r.get("datePublished",""),
                    "content":    text.strip()
                }
                try: await reviews_col.insert_one(doc); stored+=1
                except DuplicateKeyError: pass
            return {"product_id": pid, "fetched": fetched, "stored": stored}

        pr0 = (
            state0
            .get("ratingAndReviewResponse", {})
            .get("ratingAndReview", {})
            .get("productReviews", {})
        )
        if isinstance(pr0, list):
            total_pages = 1
            size = len(pr0)
            first_page_reviews = pr0
        else:
            total_pages = pr0.get("totalPages") or 1
            size        = pr0.get("size") or len(pr0.get("content", [])) or 30
            first_page_reviews = pr0.get("content", [])

        fetched += len(first_page_reviews)
        for r in first_page_reviews:
            text = r.get("comment") or r.get("reviewBody","")
            if not text: continue
            doc = {
                "product_id": pid,
                "user":       r.get("userFullName") or r.get("author",{}).get("name",""),
                "rating":     r.get("rate") or r.get("reviewRating",{}).get("ratingValue"),
                "date":       r.get("commentDateISOtype") or r.get("datePublished",""),
                "content":    text.strip()
            }
            try: await reviews_col.insert_one(doc); stored+=1
            except DuplicateKeyError: pass

        for p in range(1, total_pages):
            pp = {**base_params, "page": p, "size": size}
            rp = await client.get(api_url, params=pp); rp.raise_for_status()
            pl = rp.json()
            hs = pl["result"]["hydrateScript"]
            m1 = re.search(r"window\.__REVIEW_APP_INITIAL_STATE__\s*=\s*({.*?});", hs, re.S)
            if not m1: continue
            st = json.loads(m1.group(1))

            if isinstance(st, list):
                reviews = st
            else:
                reviews = (
                    st
                    .get("ratingAndReviewResponse", {})
                    .get("ratingAndReview", {})
                    .get("productReviews", {})
                    .get("content", [])
                )
                if not isinstance(reviews, list):
                    reviews = []

            fetched += len(reviews)
            for r in reviews:
                text = r.get("comment") or r.get("reviewBody","")
                if not text: continue
                doc = {
                    "product_id": pid,
                    "user":       r.get("userFullName") or r.get("author",{}).get("name",""),
                    "rating":     r.get("rate") or r.get("reviewRating",{}).get("ratingValue"),
                    "date":       r.get("commentDateISOtype") or r.get("datePublished",""),
                    "content":    text.strip()
                }
                try: await reviews_col.insert_one(doc); stored+=1
                except DuplicateKeyError: pass

    return {"product_id": pid, "fetched": fetched, "stored": stored}

# ----------------------------
#  Analiz uç noktası
# ----------------------------
@app.get("/analyze_by_url", response_model=AnalysisResponse)
async def analyze_by_url(
    url: str = Query(..., description="Ürün URL"),
    limit: int = Query(200, gt=0, le=500),
    offset: int = Query(0, ge=0)
):
    pid = extract_product_id(url)

    # — Genel özet için tüm yorumlar —
    all_recs  = await reviews_col.find({"product_id": pid}).to_list(length=None)
    total     = len(all_recs)
    if total == 0:
        raise HTTPException(404, "Yorum bulunamadı")

    all_texts = [r["content"] for r in all_recs]
    all_preds = await analyze_sentiments(all_texts)

    pos_all = neg_all = neut_all = 0
    complaint_all: Dict[str,int] = {}
    entity_all:    Dict[str,int] = {}

    for r, raw in zip(all_recs, all_preds):
        txt = r["content"]
        human = classify_sentiment(txt, raw)
        if human == "Olumlu":
            pos_all += 1
        elif human == "Olumsuz":
            neg_all += 1
        else:
            neut_all += 1

        # entity sayımı
        for e in ner_extractor(txt):
            entity_all[e["entity"]] = entity_all.get(e["entity"],0) + 1

        # şikayet kelime sayımı
        for kw in NEGATIVE_LEXICON:
            if re.search(rf"\b{kw}\b", txt, re.IGNORECASE):
                complaint_all[kw] = complaint_all.get(kw,0) + 1

    # — Sayfalı detay için sadece offset..offset+limit —
    recs_page = await reviews_col.find({"product_id": pid}) \
                                  .skip(offset).limit(limit) \
                                  .to_list(length=None)
    page_texts = [r["content"] for r in recs_page]
    page_preds = await analyze_sentiments(page_texts)

    ACTION_MAP = {
        "ağır":       "Ürünün ambalaj ağırlığını gözden geçirin",
        "beklemeyin": "Teslimat süresini hızlandırmayı düşünün",
        "yazık":      "Ürün kalitesini tekrar değerlendirin",
        "kötü":       "Malzeme kalitesini iyileştirin",
        "berbat":     "Ürün formülasyonunu yeniden gözden geçirin",
        "pişman":     "Müşteri beklentilerini karşılayacak iyileştirmeler yapın",
    }
    def is_true_negative(text: str, kw: str) -> bool:
        neg_pattern = rf"\b{kw}\s+(yapmıyor|değil|değildir)\b"
        return not re.search(neg_pattern, text, re.IGNORECASE)

    analyses: List[Analysis] = []
    for r, raw in zip(recs_page, page_preds):
        text = r["content"]
        human = classify_sentiment(text, raw)

        # somut aksiyon önerileri
        triggered_actions: Set[str] = set()
        for kw, act in ACTION_MAP.items():
            if re.search(rf"\b{kw}\b", text, re.IGNORECASE) and is_true_negative(text, kw):
                triggered_actions.add(act)

        ents = ner_extractor(text)
        sugg: List[str] = []
        if human == "Olumlu":  sugg.append("Yorumlar olumlu")
        if human == "Olumsuz": sugg.append("Olumsuz yorum tespit edildi")
        for act in triggered_actions: sugg.append(act)
        if any(e["entity"]=="SIZE_COMMENT" for e in ents): sugg.append("Beden tavsiyesi var")
        if any(e["entity"]=="RECIPIENT"    for e in ents): sugg.append("Alıcıya hediye edilmiş")
        if any(e["entity"]=="OCCASION"     for e in ents): sugg.append("Kampanya dönemi yorumu")

        analyses.append(Analysis(
            text=text,
            sentiment={"label": human, "score": raw["score"]},
            entities=ents,
            suggestion="; ".join(sugg) or "—"
        ))

    # — Summary oluşturma —
    def pct(x): return round(100 * x / total, 1)
    top_complaints = sorted(complaint_all.items(), key=lambda x: -x[1])[:3]
    comp_phrases = [f"{cnt} kez “{kw}”" for kw,cnt in top_complaints] or ["Belirgin şikâyet yok"]
    top_entities = sorted(entity_all.items(), key=lambda x: -x[1])[:2]
    ent_phrases  = [f"{name} ({cnt})" for name,cnt in top_entities] or ["Öne çıkan konu etiketi yok"]
    overall = "olumlu" if pos_all >= neg_all else "olumsuz"
    summary = (
        f"Toplam {total} yorum: %{pct(pos_all)} olumlu, %{pct(neg_all)} olumsuz, %{pct(neut_all)} nötr. "
        f"En çok şikayet edilenler: {', '.join(comp_phrases)}. "
        f"Ayrıca konu etiketleri: {', '.join(ent_phrases)}. "
        f"Genel olarak kullanıcılar ürünü “{overall}” bulmuş."
    )

    return AnalysisResponse(
        product_id=pid,
        total=total,
        limit=limit,
        offset=offset,
        analyses=analyses,
        summary=summary
    )

# ----------------------------
#  Run with:
#    python -m uvicorn main:app --reload
# ----------------------------