# image_generator.py
from transformers import MarianMTModel, MarianTokenizer, CLIPTokenizer
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os
import datetime
import logging
from config_loader import device

# Logger yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_translation_models(model_name):
    """Belirtilen model ismiyle çeviri modellerini döndürür."""
    try:
        translation_model = MarianMTModel.from_pretrained(model_name).to(device)
        translation_tokenizer = MarianTokenizer.from_pretrained(model_name)
        logger.info(f"Çeviri modeli yüklendi: {model_name}")
        return translation_model, translation_tokenizer
    except Exception as e:
        logger.error(f"Çeviri modeli yüklenirken hata oluştu: {e}")
        return None, None

def get_image_pipeline(model_id):
    """Belirtilen model ismiyle görsel oluşturma pipeline'ı döndürür."""
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        ).to(device)
        logger.info(f"Görsel oluşturma pipeline'ı yüklendi: {model_id}")
        return pipe
    except Exception as e:
        logger.error(f"Görsel oluşturma pipeline'ı yüklenirken hata oluştu: {e}")
        return None

# CLIP tokenizer'ı yükleyin
try:
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    logger.info("CLIP Tokenizer yüklendi.")
except Exception as e:
    logger.error(f"CLIP Tokenizer yüklenirken hata oluştu: {e}")
    clip_tokenizer = None

def translate_text(text, translation_model, translation_tokenizer):
    if not translation_model or not translation_tokenizer:
        logger.error("Çeviri modeli veya tokenizer mevcut değil.")
        return text

    try:
        # Maksimum giriş uzunluğunu belirleyin
        max_input_length = 512  # Manuel olarak ayarlayın
        
        # Tokenizer'ı kullanarak girişleri hazırlayın
        inputs = translation_tokenizer(
            text,
            return_tensors='pt',
            max_length=max_input_length,
            truncation=True,
            padding='max_length'
        ).to(device)

        # Çeviri modelini kullanarak çıktı alın
        outputs = translation_model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=512,  # Çıktı için maksimum uzunluk
            num_beams=5,
            early_stopping=True
        )

        # Çıktıyı metne dönüştürün
        english_summary = translation_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        logger.info("Çeviri başarılı.")
        return english_summary

    except Exception as e:
        logger.error(f"Çeviri sırasında hata oluştu: {e}")
        return text  # Hata durumunda orijinal metni döndür

def generate_image(english_summary, title, pipe, model_name):
    """ İngilizce özet üzerinden görsel üretir ve kaydeder. """
    if not english_summary:
        logger.error("Görsel oluşturmak için geçerli bir metin yok.")
        return None

    if not clip_tokenizer:
        logger.error("CLIP Tokenizer mevcut değil.")
        return None

    # Prompt'u CLIP'in 77 token sınırına göre kes
    base_prompt = "A happy watercolor illustration of a children's fairy tale"
    base_tokens = clip_tokenizer.tokenize(base_prompt)
    max_total_length = 77  # CLIP'in maksimum token uzunluğu
    max_prompt_length = max_total_length - len(base_tokens) - 2  # BOS ve EOS tokenleri için -2

    tokens = clip_tokenizer.tokenize(english_summary)
    if len(tokens) > max_prompt_length:
        tokens = tokens[:max_prompt_length]
        english_summary = clip_tokenizer.convert_tokens_to_string(tokens)

    prompt = f"{base_prompt}{english_summary}"

    try:
        if not pipe:
            logger.error("Görsel oluşturma pipeline'ı mevcut değil.")
            return None

        image = pipe(prompt).images[0]
        static_folder = os.path.join('static', 'images')  # 'static/images' klasörü
        os.makedirs(static_folder, exist_ok=True)
        
        # UTC zaman damgasını al
        utc_time = datetime.datetime.utcnow()
        timestamp = utc_time.strftime("%Y%m%d%H%M%S")
        
        # Model ismini dosya adına ekle (güvenli hale getirmek için boşlukları alt çizgiyle değiştir)
        safe_model_name = model_name.replace('/', '_').replace('-', '_')

        image_filename = f"{title}_{safe_model_name}_{timestamp}.png"
        image_path = os.path.join(static_folder, image_filename)
        image.save(image_path)
        logger.info(f"Görsel '{image_filename}' başarıyla kaydedildi.")
        # Görsel yolunu URL dostu hale getir
        image_url = os.path.join('images', image_filename).replace('\\', '/')
        return image_url
    except Exception as e:
        logger.error(f"Görsel oluşturma sırasında hata oluştu: {e}")
        return None
