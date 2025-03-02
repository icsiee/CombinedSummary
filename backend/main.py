from fastapi import FastAPI
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import torch

# FastAPI uygulamasını oluştur
app = FastAPI()

# CORS ayarları (React uygulaması için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React uygulamasının URL'si
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cihaz kontrolü (GPU varsa kullan)
device = 0 if torch.cuda.is_available() else -1
print(f"Device set to use: {'CUDA (GPU)' if device == 0 else 'CPU'}")

# BERT modeli ile konu analizi
classifier = pipeline(
    "text-classification",
    model="dbmdz/bert-base-turkish-cased",  # Türkçe metinler için BERT modeli
    device=device,
)

# Özetleme için BART modeli
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",  # Özetleme için BART modeli
    device=device,
)

# Örnek haber metinleri
sample_news = [
    "29 Mayıs 1453'te Osmanlı Padişahı II. Mehmet önderliğinde İstanbul fethedildi. Bizans İmparatorluğu'nun başkenti olan şehir, Osmanlı topraklarına katıldı. Kuşatma sırasında kullanılan büyük toplar, surların yıkılmasında etkili oldu. İstanbul'un fethi, Orta Çağ’ın sonu ve Yeni Çağ’ın başlangıcı olarak kabul edilir. II. Mehmet, bu zaferden sonra unvanını aldı. Osmanlı Devleti’nin başkenti Bursa’dan İstanbul’a taşındı. Şehrin kültürel ve ekonomik yapısında büyük değişimler yaşandı. Ayasofya, camiye dönüştürülerek ibadete açıldı. İstanbul, Osmanlı’nın en önemli ticaret ve kültür merkezi haline geldi. Fetih, Osmanlı’nın dünya sahnesindeki gücünü artırdı.",
    "Her yıl 29 Mayıs’ta İstanbul’un fethi çeşitli etkinliklerle anılıyor. Bu yıl da İstanbul’da büyük kutlamalar yapıldı. Törenler, Mehter Takımı’nın gösterisiyle başladı. İstanbul’un fethi, Türk tarihinde bir dönüm noktası olarak kabul ediliyor. Kutlamalar kapsamında tarihi canlandırmalar gerçekleştirildi. Fatih Sultan Mehmet’in şehre girişini temsilen tiyatro gösterisi düzenlendi. Ayasofya ve Topkapı Sarayı’nda özel sergiler açıldı. Fetih, Osmanlı Devleti’nin yükseliş döneminin başlangıcı oldu. Cumhurbaşkanı, törende yaptığı konuşmada fetih ruhunun önemine değindi. Kutlamalara binlerce kişi katılarak coşkuyla İstanbul’un fethini andı.",
    "İstanbul’un fethi, Bizans İmparatorluğu’nun sonunu getirdi. II. Mehmet’in komutasındaki Osmanlı ordusu, 53 gün süren kuşatmayı başarıyla tamamladı. Osmanlı ordusu, Haliç’e indirilen gemilerle büyük bir avantaj sağladı. 29 Mayıs 1453’te Osmanlı askerleri Topkapı surlarından şehre girdi. Bizans İmparatoru XI. Konstantinos, savaş meydanında hayatını kaybetti. Fetihle birlikte Osmanlı, Avrupa ve Asya arasında köprü kurdu. İstanbul, Osmanlı’nın yeni başkenti olarak ilan edildi. Fetihten sonra şehirde çeşitli imar çalışmaları başlatıldı. Osmanlı mimarisiyle yeniden şekillenen İstanbul, İslam dünyasının önemli merkezlerinden biri oldu. İstanbul’un fethi, dünya tarihindeki en önemli olaylardan biri olarak kabul edilir.",
    "Fatih Sultan Mehmet, İstanbul’un fethi için uzun süre hazırlık yaptı. Osmanlı ordusu, fetih öncesinde yeni toplar üreterek güçlendi. Rumeli Hisarı’nın inşası, kuşatmanın başarısında önemli rol oynadı. Osmanlı donanması, Haliç’e karadan gemi indirerek Bizans’ı şaşırttı. 53 gün süren kuşatma sonunda surlar aşıldı. Fetihle birlikte Osmanlı Devleti, büyük bir güce ulaştı. İstanbul, Osmanlı Devleti’nin siyasi ve ekonomik merkezi oldu. Fatih Sultan Mehmet, şehirde birçok reform gerçekleştirdi. İstanbul’un fethi, Osmanlı’nın Avrupa’daki ilerleyişini hızlandırdı. Fatih’in stratejik zekâsı, Osmanlı tarihindeki en büyük zaferlerden birini getirdi.",
]

# Metni parçalara bölme fonksiyonu
def split_text(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk)) + len(word) + 1 <= max_length:
            current_chunk.append(word)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Haber özetleme fonksiyonu
def generate_summary(text):
    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        summaries.append(summary)

    combined_summary = " ".join(summaries)
    return combined_summary

# API endpoint: Tüm haberler için tek bir özet oluştur
@app.get("/get_combined_summary")
def get_combined_summary():
    combined_news = " ".join(sample_news)
    summary = generate_summary(combined_news)
    return {"combined_summary": summary}

# Ana sayfa
@app.get("/")
def read_root():
    return {"message": "Welcome to the News Summarization API!"}