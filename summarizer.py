#summarizer.py
import nltk
import re
import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, PegasusForConditionalGeneration, MT5Tokenizer, MT5ForConditionalGeneration, BartForConditionalGeneration, BertTokenizerFast, EncoderDecoderModel
from config_loader import device

# Logger'ı alıyoruz (app.py'da yapılandırıldı)
logger = logging.getLogger(__name__)

# Gerekli NLTK verilerini kontrol et ve indir
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sections_sentence_based(text, logger):
    logger.info("Metin bölümlere ayrılıyor...")
    """Metni giriş (%30), gelişme (%40) ve sonuç (%30) olarak böler."""
    sentences = nltk.sent_tokenize(text)
    total = len(sentences)

    if total < 4:
        # Çok kısa metinlerde eşit böl
        return ' '.join(sentences[:1]), ' '.join(sentences[1:2]), ' '.join(sentences[2:])

    # Yüzdelere göre böl
    intro_end = 3 * (total // 10)
    dev_end = intro_end + 4 * (total // 10)

    intro = sentences[:intro_end]  # İlk %30
    dev = sentences[intro_end:dev_end]  # Orta %40
    conc = sentences[dev_end:]  # Son %30

    return ' '.join(intro), ' '.join(dev), ' '.join(conc)

def get_summarizer_pipeline(model_name, device, logger):
    """Belirtilen model ismiyle bir özetleme (veya text-generation) pipeline'ı döndürür."""
    try:
        # Özel modeller için koşullar
        if model_name.startswith("nebiberke/news"):
            tokenizer = MT5Tokenizer.from_pretrained(model_name)
            model = MT5ForConditionalGeneration.from_pretrained(model_name)

        elif model_name.startswith("google/pegasus"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name)

        elif model_name.startswith("sshleifer/distilbart"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)

        elif model_name.startswith("mrm8488/bert2bert"):
            tokenizer = BertTokenizerFast.from_pretrained(model_name)
            model = EncoderDecoderModel.from_pretrained(model_name).to(device)

        else:
            # Varsayılan yükleme
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # Cihaz seçimi
        model = model.to(device)

        # Modelin konfigürasyonundan maksimum girdi uzunluğunu al
        config = model.config
        
        max_input_length = getattr(config, 'max_position_embeddings', 1024)

        logger.info(f"{model_name} modeli için maksimum girdi uzunluğu: {max_input_length}")

        # Pipeline'ı oluştur
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.type == "cuda" else -1,
            clean_up_tokenization_spaces=True,
        )
        logger.info(f"Özetleme pipeline'ı yüklendi: {model_name}")
        return summarizer, tokenizer, max_input_length

    except Exception as e:
        logger.error(f"Özetleme pipeline'ı yüklenirken hata oluştu: {e}")
        return None, None, None



def summarize_text(text, summarizer_pipeline, tokenizer, max_input_length, logger):
    """
    Metni (intro, dev, conc) olarak ayır ve her bölümü chunking ile özetle.
    """
    # 1) Metni 3 parçaya ayıran fonksiyon:
    intro_text, dev_text, conc_text = split_into_sections_sentence_based(text, logger)
    
    logger.info("Metin giriş, gelişme ve sonuç bölümlerine ayrıldı.")

    # 2) Bölümleri özetle
    introduction = summarize_section(intro_text, summarizer_pipeline, tokenizer, logger, 'introduction')
    development = summarize_section(dev_text, summarizer_pipeline, tokenizer, logger, 'development')
    conclusion = summarize_section(conc_text, summarizer_pipeline, tokenizer, logger, 'conclusion')

    return introduction, development, conclusion


def summarize_section(section_text, summarizer_pipeline, tokenizer, logger, section):
    """
    Tek bir bölümü chunking ile özetler.
    """
    # Token sayısı
    tokens = tokenizer.encode(section_text, return_tensors='pt')
    token_length = tokens.size(1)
    logger.info(f"{section.capitalize()} bölümünün girdi token sayısı: {token_length}")

    # Belirli bir chunk_size eşiği (örn. 800 veya 1024)
    chunk_size = 800
    if token_length <= chunk_size:
        # Kısa ise direkt özetle
        return _summarize_chunk(section_text, summarizer_pipeline, tokenizer, logger, section)
    else:
        # Uzun ise parçalara böl
        all_tokens = tokens[0]
        chunk_summaries = []
        for start_idx in range(0, token_length, chunk_size):
            end_idx = start_idx + chunk_size
            chunk_slice = all_tokens[start_idx:end_idx]
            chunk_text = tokenizer.decode(chunk_slice, skip_special_tokens=True)
            
            # Parça özet
            partial_summary = _summarize_chunk(
                chunk_text, summarizer_pipeline, tokenizer, logger, f"{section}_partial"
            )
            chunk_summaries.append(partial_summary)

        # Parça özetlerini birleştir
        combined = " ".join(chunk_summaries)

        return combined


def _summarize_chunk(text, summarizer_pipeline, tokenizer, logger, section):
    try:
        tokens = tokenizer.encode(text, return_tensors='pt')
        token_length = tokens.size(1)

        if section == "introduction":
            max_len = min(250, max(50, int(0.3 * token_length)))
            min_len = max(50, int(0.15 * token_length))
        elif section == "development":
            max_len = min(300, max(120, int(0.4 * token_length)))
            min_len = max(100, int(0.2 * token_length))
        else:  # conclusion
            max_len = min(250, max(50, int(0.3 * token_length)))
            min_len = max(50, int(0.15 * token_length))

        if min_len >= max_len:
            min_len = max_len - 1

        # Logger bilgilerini doğru sırada ekle
        logger.info(f"{section.capitalize()} chunk token sayısı: {token_length}")
        logger.info(f"{section.capitalize()} chunk max token sayısı: {max_len}")
        logger.info(f"{section.capitalize()} chunk min token sayısı: {min_len}")

        # Özetleme çağrısı
        summary = summarizer_pipeline(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            num_beams=6,
            repetition_penalty=1.4
        )[0]['summary_text']

        summary_tokens = tokenizer.encode(summary, return_tensors='pt').size(1)
        logger.info(f"{section.capitalize()} özet token sayısı: {summary_tokens}")

        return summary

    except Exception as e:
        logger.error(f"Error in summarizing chunk: {e}")
        return text
