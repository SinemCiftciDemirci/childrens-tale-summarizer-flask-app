import os
from tkinter import Tk, filedialog
from transformers import MarianMTModel, MarianTokenizer, CLIPTokenizer
from image_generator import get_translation_models, get_image_pipeline, translate_text
from config_loader import device

# Çeviri ve görsel oluşturma modellerini tanımlayın
translation_model_name = "Helsinki-NLP/opus-mt-tr-en"  # Türkçe'den İngilizce'ye çeviri modeli
image_model_name = "runwayml/stable-diffusion-v1-5"    # Görsel oluşturma modeli
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Modelleri yükleyin
translation_model, translation_tokenizer = get_translation_models(translation_model_name)
pipe = get_image_pipeline(image_model_name)

# Kullanıcıdan dosya seçmesini isteyin
def select_file():
    Tk().withdraw()  # Tkinter GUI'yi gizle
    file_path = filedialog.askopenfilename(
        title="Bir TXT dosyası seçin",
        filetypes=[("Text Files", "*.txt")]
    )
    return file_path

# Giriş, Gelişme ve Sonuç bölümlerini işleme
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    sections = {"Giriş": None, "Gelişme": None, "Sonuç": None}
    for section in sections:
        start_idx = content.find(f"{section}:")
        if start_idx != -1:
            end_idx = content.find("\n\n", start_idx)
            end_idx = end_idx if end_idx != -1 else len(content)
            sections[section] = content[start_idx + len(section) + 1:end_idx].strip()

    return sections

# Görsel oluşturmayı CLIP prompt sınırıyla entegre eden fonksiyon
def generate_image_with_prompt(english_summary, base_prompt, pipe, output_folder, title):
    try:
        # CLIP prompt sınırı
        base_tokens = clip_tokenizer.tokenize(base_prompt)
        max_total_length = 77  # CLIP'in maksimum token uzunluğu
        max_prompt_length = max_total_length - len(base_tokens) - 2  # BOS ve EOS tokenleri için -2

        tokens = clip_tokenizer.tokenize(english_summary)
        if len(tokens) > max_prompt_length:
            tokens = tokens[:max_prompt_length]
            english_summary = clip_tokenizer.convert_tokens_to_string(tokens)

        prompt = f"{base_prompt} {english_summary}"
        
        # Görsel oluşturma
        image = pipe(prompt).images[0]
        os.makedirs(output_folder, exist_ok=True)
        image_path = os.path.join(output_folder, f"{title}.png")
        image.save(image_path)
        print(f"Görsel '{image_path}' kaydedildi.")
        return image_path
    except Exception as e:
        print(f"Görsel oluşturulamadı: {e}")
        return "Görsel oluşturulamadı."

# Çıktıyı kaydet
def save_results(output_folder, translations, image_paths):
    os.makedirs(output_folder, exist_ok=True)  # Klasörü oluştur

    output_file = os.path.join(output_folder, "results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for section, translated_text in translations.items():
            f.write(f"{section}:\n{translated_text}\n")
            f.write(f"Görsel: {image_paths.get(section, 'Görsel oluşturulamadı.')}\n\n")
    print(f"Sonuçlar ve görseller '{output_folder}' klasörüne kaydedildi.")

# Ana işlem
def main():
    file_path = select_file()
    if not file_path:
        print("Dosya seçilmedi.")
        return

    print(f"Seçilen dosya: {file_path}")
    sections = process_file(file_path)

    output_folder = "translate_vision_test"
    base_prompt = "A watercolor illustration of a children story"
    translations = {}
    image_paths = {}

    for section, text in sections.items():
        if text:
            print(f"{section} bölümü işleniyor...")
            # Metni çevir
            translated_text = translate_text(text, translation_model, translation_tokenizer)
            translations[section] = translated_text

            # Görsel oluştur
            image_path = generate_image_with_prompt(translated_text, base_prompt, pipe, output_folder, section)
            image_paths[section] = image_path

    # Çeviriler ve görsel yolları kaydediliyor
    save_results(output_folder, translations, image_paths)

if __name__ == "__main__":
    main()
