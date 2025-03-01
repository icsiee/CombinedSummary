from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
import torch
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# BERT summarizer pipeline (GPU kullanımı için)
device = 0 if torch.cuda.is_available() else -1  # GPU varsa, 0 kullan, yoksa CPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)  # Büyük BART modelini kullandık

# Sample news text
sample_news = [
    " 2. Dünya Savaşı, 1939 yılında Almanya'nın Polonya'ya saldırmasıyla başladı. Bu saldırı, İngiltere ve Fransa'nın Almanya'ya savaş ilan etmesine yol açtı. Savaşın hemen ardından Almanya, Sovyetler Birliği ile Molotov-Ribbentrop Paktı'nı imzalayarak Avrupa'da stratejik bir denge kurmayı hedefledi. Ancak, 1941'de Almanya'nın Sovyetler Birliği'ne saldırması savaşın kaderini değiştirdi. Amerika Birleşik Devletleri'nin Pearl Harbor'a yapılan Japon saldırısının ardından savaşa dahil olması, müttefiklerin lehine bir dönüm noktası oldu. Almanya'nın Batı cephesinde müttefiklerle karşılaştığı çetin direniş, Doğu cephesinde Sovyetlerin ilerlemesiyle birleşti. Normandiya Çıkarması, müttefiklerin Avrupa'ya girmesinin başlangıcıydı. 1945 yılında Sovyetler Birliği Berlin'i işgal etti ve Nazi Almanyası teslim oldu. Japonya ise atom bombasıyla karşı karşıya kaldı. Sonunda, 2. Dünya Savaşı 1945'te sona erdi ve dünya yeniden şekillendi.",
    " 2. Dünya Savaşı, tarihteki en büyük çatışmalardan biri olarak kaydedildi. Almanya'nın Polonya'ya saldırısıyla başlayan savaş, tüm dünyayı etkisi altına aldı. Avrupa'da savaş, Nazizm'in yükselmesine ve Almanya'nın hızlıca toprak kazanmasına neden oldu. Ancak, Almanya'nın Sovyetler Birliği'ne saldırısı, iki büyük gücün karşı karşıya gelmesine yol açtı. Japonya'nın Asya'daki genişleme arzusu, Pasifik'teki savaşın alevlenmesine neden oldu. 1941'de Amerika Birleşik Devletleri'nin savaşa katılması, müttefiklerin stratejik üstünlük kazanmasına yardımcı oldu. 1944 yılında Normandiya Çıkarması ile Batı Avrupa'da büyük bir ilerleme kaydedildi. 1945'te Sovyetler Birliği, Berlin'e girerek Nazi Almanyası'nı sona erdirdi. Aynı yıl Japonya, atom bombalarının ardından teslim oldu. Bu savaş, dünya siyasetini ve sınırlarını köklü şekilde değiştirdi.",
    " 2. Dünya Savaşı, sadece Avrupa'yı değil, tüm dünyayı etkileyen bir savaş haline geldi. 1939'da Almanya'nın Polonya'ya saldırmasıyla başlayan savaş, kısa sürede tüm kıtalara yayıldı. Batı Avrupa'da, Almanya hızla Fransız topraklarını işgal etti. Sovyetler Birliği ile yapılan anlaşmaların ardından, Almanya'nın Sovyetler Birliği'ne saldırması savaşın büyük bir dönüm noktasıydı. Japonya, Asya'da toprak kazanarak Pasifik'teki savaşın önemli bir oyuncusu oldu. Amerika Birleşik Devletleri, Japonya'nın Pearl Harbor'a saldırmasının ardından savaşa katıldı. Normandiya Çıkarması ile Batı Avrupa'da savaşın seyrini değiştiren müttefikler, Sovyetler Birliği ile işbirliği yaptı. 1945'te Almanya teslim oldu ve Avrupa'da savaş sona erdi. Japonya ise atom bombalarının etkisiyle teslim oldu. Bu savaşın sonunda dünya siyasi haritası yeniden çizildi.",
    " 2. Dünya Savaşı, dünya tarihinin en büyük askeri çatışması olarak kabul edilmektedir. 1939'da Almanya'nın Polonya'ya saldırmasıyla başlayan savaş, kısa süre içinde büyük bir küresel çatışmaya dönüştü. Almanya, Avrupa'da hızlı bir şekilde toprak kazandı ve İngiltere ile Fransız direnişleriyle karşılaştı. 1941'de Sovyetler Birliği'ne yapılan saldırı, savaşın doğu cephesini başlattı. Japonya, Asya'da genişleme politikası izleyerek Pasifik Okyanusu'na hakim olmayı hedefledi. 1941'de Amerika Birleşik Devletleri'nin savaşa katılması, müttefiklerin gücünü artırdı. 1944 yılında Normandiya Çıkarması ile Batı Avrupa'da savaşın seyrini değiştiren müttefikler, Berlin'e kadar ilerledi. 1945'te Sovyetler Berlin'i işgal etti ve Nazi Almanyası teslim oldu. Japonya ise atom bombaları sonrası teslim oldu. 2. Dünya Savaşı, dünya genelindeki siyasi dengeleri değiştirdi ve yeni bir dünya düzeninin temellerini attı.",
    " 2. Dünya Savaşı, 1939 yılında Almanya'nın Polonya'ya saldırmasıyla başladı. Bu saldırı, savaşın patlak vermesine neden oldu ve İngiltere ile Fransa Almanya'ya savaş ilan etti. Almanya, hızla Avrupa'da toprak kazandı ve Sovyetler Birliği'ne karşı büyük bir saldırı başlattı. 1941'de Amerika Birleşik Devletleri'nin savaşa katılması, müttefiklerin lehine bir gelişme oldu. Japonya'nın Asya'da genişleme arzusu, Pasifik'teki savaşın büyümesine yol açtı. Normandiya Çıkarması, Batı Avrupa'da büyük bir direnişi kırarak müttefiklerin zafer kazanmasına yardımcı oldu. 1945 yılında Sovyetler Birliği Berlin'i işgal etti ve Nazi Almanyası teslim oldu. Japonya ise atom bombaları nedeniyle teslim oldu. Savaş, dünya siyasetinde köklü değişikliklere yol açtı. Sonunda 2. Dünya Savaşı sona erdi ve dünya yeni bir düzene girdi."

]

# Function to generate a summary of each news in parallel
def generate_summary(news):
    sentences = news.split(". ")
    truncated_news = ". ".join(sentences[:15])  # Daha fazla cümle kullanalım
    input_length = len(truncated_news.split())
    max_length = max(input_length // 2, 100)  # Minimum 100 kelime ile özet
    summary = summarizer(truncated_news, max_length=max_length, min_length=75, do_sample=False)  # Min_length arttırıldı
    return summary[0]['summary_text']

@app.get("/get_sample_news")
def get_sample_news():
    # Use ThreadPoolExecutor for parallel processing of news summaries
    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(generate_summary, sample_news))

    # Now summarize the summaries
    full_summary = " ".join(summaries)
    sentences = full_summary.split(". ")  # Split the final summaries into sentences
    truncated_full_summary = ". ".join(sentences[:5])  # İlk 5 cümleyi alalım
    
    # Calculate the input length for the final summary
    input_length = len(truncated_full_summary.split())
    max_length = max(input_length // 2, 100)  # Minimum 100 kelime ile özet

    # Generate the final summary
    final_summary = summarizer(truncated_full_summary, max_length=max_length, min_length=75, do_sample=False)  # Min_length arttırıldı

    # Check if the final summary ends with a complete sentence
    final_text = final_summary[0]['summary_text']
    if not final_text.endswith("."):
        # If it doesn't end with a period, try to append the last part to make it a complete sentence
        final_text += "."
    
    return {
        "final_summary": final_text
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the News Summarization API!"}
