from sentence_transformers import SentenceTransformer, util
import nltk
import tkinter as tk
from tkinter import filedialog
import os

# Gerekli NLTK verilerini kontrol et ve indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Dosya aç
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Bir Metin Dosyası Seçin", filetypes=[("Metin Dosyaları", "*.txt")])

if not file_path:
    print("Dosya seçilmedi.")
    exit()

# Dosya oku
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Metni cümlelere böl
sentences = nltk.sent_tokenize(text)
total = len(sentences)
print(total)

# Modeli yükle (BERT tabanlı bir Sentence Transformer modeli)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Tüm cümleleri vektörize et
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Metnin tamamını bir vektöre çevir
text_embedding = model.encode(text, convert_to_tensor=True)

# Cosine similarity hesapla
similarities = util.pytorch_cos_sim(sentence_embeddings, text_embedding)

# Benzerlik skorlarına göre sıralama
similarity_scores = similarities.squeeze(1).tolist()
ranked_sentences = sorted(zip(similarity_scores, range(len(sentences))), reverse=True)

# En yüksek skora sahip cümlelerin orijinal sıraya göre dizilmesi
num_sentences = total // 6
print(num_sentences)
selected_sentences_indices = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])

# Orijinal sıralamaya göre özet
summary_sentences = [sentences[index] for _, index in selected_sentences_indices]
reference_summary = " ".join(summary_sentences)

# Referans özet
#print("Referans Özet:")
#print(reference_summary)

# Özeti 'cos_sim_summaries' dosyasına kaydet
summaries_folder = "cos_sim_summaries"
if not os.path.exists(summaries_folder):
    os.makedirs(summaries_folder)

# Dosya adına göre yeni dosya adı oluşturma
base_name = os.path.basename(file_path)
summary_file_path = os.path.join(summaries_folder, f"{os.path.splitext(base_name)[0]}.txt")

with open(summary_file_path, "w", encoding="utf-8") as summary_file:
    summary_file.write(reference_summary)
        
print(f"Özet '{summary_file_path}' konumuna kaydedildi.")
