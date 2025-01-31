import openai
import tkinter as tk
from tkinter import filedialog
import os
from config_loader import load_keys_from_file

# Config dosyasını yükle
config = load_keys_from_file("config.txt")

# OpenAI API anahtarını al
api_key = config.get("OPENAI_API_KEY")

def upload_and_summarize(api_key):
    """
    Select a file from the file explorer and summarize its content
    into three sections: Introduction, Development, and Conclusion in Turkish,
    and save the summary in the 'gpt_summaries' folder.
    
    Args:
        api_key (str): OpenAI API key
    """
    # OpenAI API key
    openai.api_key = api_key

    # Dosya Seç
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Bir Metin Dosyası Seçin", filetypes=[("Metin Dosyaları", "*.txt")])

    if not file_path:
        print("Dosya seçilmedi.")
        return

    # Dosyayı oku
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    # Türkçe özet için prompt
    prompt = (
    "Bir masal özeti yazmanı istiyorum. Özet, aşağıdaki kurallara uygun olmalı:\n"
    "1. Ekstraktif özet istiyorum. Özet, tamamen verilen hikayeye bağlı kalmalı. Ek bilgi eklememeli veya hikaye dışı yaratıcı ifadeler kullanmamalı.\n"
    "2. Metindeki olayları değiştirmemeli, olduğu gibi anlatmalı.\n"
    "3. Anlatım masalsı bir üslup taşımalı ancak abartıya kaçmamalı.\n"
    "4. Detaylı diyaloglar ve karakterlerin doğrudan tepkileri metinden alınarak kullanılmalı.\n"
    "5. 'Yaptı', 'etti' yerine 'yapmış', 'etmiş' gibi ifadeler kullanılmalı.\n"
    "6. Klişe ifadeler (örneğin, 'Masal burada biter' veya 'Ve böylece macera sona erdi') kullanılmamalı.\n"
    "7. Anlatım, hikayedeki sıralamayı ve olay örgüsünü korumalı. Gereksiz çıkarımlar yapılmamalı.\n\n"
    "Hikayeyi üç ana bölüme ayır: Giriş (%30), Gelişme (%40) ve Sonuç (%30). "
    "Her bölümü açıkça 'Giriş:', 'Gelişme:', 'Sonuç:' başlıkları altında sun. "
    "Toplam uzunluk hikayenin 1/6'sı kadar olsun.\n\n"
    f"Hikaye:\n{content}"
)

    # OpenAI API çağrısı
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sen hikayeleri Türkçe olarak masalsı bir dille özetleyen bir asistansın."},
                {"role": "user", "content": prompt}
            ]
        )

        summary = response['choices'][0]['message']['content']
        
        # Özeti 'gpt_summaries' klasörüne kaydet
        summaries_folder = "gpt_summaries"
        if not os.path.exists(summaries_folder):
            os.makedirs(summaries_folder)

        # Dosya adına göre yeni dosya adı oluştur
        base_name = os.path.basename(file_path)
        summary_file_path = os.path.join(summaries_folder, f"{os.path.splitext(base_name)[0]}.txt")

        with open(summary_file_path, "w", encoding="utf-8") as summary_file:
            summary_file.write(summary)
        
        print(f"Özet '{summary_file_path}' konumuna kaydedildi.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    if api_key:
        upload_and_summarize(api_key)
    else:
        print("OpenAI API anahtarı config dosyasından alınamadı.")
