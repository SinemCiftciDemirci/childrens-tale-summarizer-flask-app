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

def calculate_lengths(token_length, section):
    if section == "introduction":
        max_len = min(300, int(0.3 * token_length))
        min_len = max(80, int(0.15 * token_length))
    elif section == "development":
        max_len = min(300, int(0.3 * token_length))
        min_len = max(90, int(0.15 * token_length))
    else:  # conclusion
        max_len = min(250, int(0.3 * token_length))
        min_len = max(80, int(0.15 * token_length))


    if min_len >= max_len:
        min_len = max_len - 1

    return max_len, min_len

# Parametreler
PARAMETERS = {
    "introduction": {
        "repetition_penalty": 1.4,
        "no_repeat_ngram_size": 2,
        "top_k": 50,
        "top_p": 0.7,
        "temperature": 0.5,
        "num_beams": 8,
        "do_sample": True
    },
    "development": {
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "top_k": 50,
        "top_p": 0.8,
        "temperature": 0.5,
        "num_beams": 16,
        "do_sample": True
    },
    "conclusion": {
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3,
        "top_k": 50,
        "top_p": 0.7,
        "temperature": 0.5,
        "num_beams": 9,
        "do_sample": True
    }
}

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

    # Eğer token bazında ölçmek isterseniz:
    intro_tokens = len(tokenizer.encode(introduction, add_special_tokens=False))
    dev_tokens = len(tokenizer.encode(development, add_special_tokens=False))
    conc_tokens = len(tokenizer.encode(conclusion, add_special_tokens=False))
    logger.info(f"Introduction: {intro_tokens} token")
    logger.info(f"Development: {dev_tokens} token")
    logger.info(f"Conclusion: {conc_tokens} token")

    return introduction, development, conclusion

def summarize_section(section_text, summarizer_pipeline, tokenizer, logger, section):
    """
    Tek bir bölümü her zaman chunking ile özetler.
    """
    # Token sayısı
    tokens = tokenizer.encode(section_text, return_tensors='pt')
    token_length = tokens.size(1)
    logger.info(f"{section.capitalize()} bölümünün girdi token sayısı: {token_length}")

    # Chunklama parametreleri
    max_len, min_len = calculate_lengths(token_length, section)
    params = PARAMETERS.get(section, {})
    if not params:
        logger.warning(f"{section} için parametreler bulunamadı. Varsayılan değerler kullanılacak.")

    chunk_size = 800  # Chunk başına maksimum token sayısı
    all_tokens = tokens[0]
    chunk_summaries = []

    # Chunk'lara ayır ve her chunk'ı özetle
    for start_idx in range(0, token_length, chunk_size):
        end_idx = start_idx + chunk_size
        chunk_slice = all_tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_slice, skip_special_tokens=True)

        summary = summarizer_pipeline(
            chunk_text,
            max_length=max_len,
            min_length=min_len,
            do_sample=params['do_sample'],
            top_k=params['top_k'],
            top_p=params['top_p'],
            temperature=params['temperature'],
            num_beams=params['num_beams'],
            repetition_penalty=params['repetition_penalty'],
            no_repeat_ngram_size=params.get("no_repeat_ngram_size", 3)
        )[0]['summary_text']

        chunk_summaries.append(summary)

    # Parça özetlerini birleştir
    combined = " ".join(chunk_summaries)
    return combined
