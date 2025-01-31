# app.py
import os
import torch
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from datetime import datetime
import PyPDF2
from sqlalchemy.sql import func
from config_loader import load_keys_from_file, device
from summarizer import summarize_text, clean_text, get_summarizer_pipeline
from image_generator import generate_image, translate_text, get_translation_models, get_image_pipeline
import warnings

# Tüm uyarıları görmezden gel
warnings.filterwarnings("ignore")

# Logging yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """
    Flask uygulamasını ve modelleri yükleyen fonksiyon.
    Windows üzerinde multiprocessing tekrar import ettiğinde
    bu fonksiyon çağrılmayacak, ancak if __name__ == '__main__': altında
    manuel olarak çağıracağız.
    """
    # Global device kullanımı
    global device  # Global cihaz değişkeni
    logger.info(f"Using global device: {device}")

    # Anahtarları config.txt dosyasından yükleyin
    keys = load_keys_from_file('config.txt')
    
    app = Flask(__name__)
    app.secret_key = keys.get('SECRET_KEY')
    
    if not app.secret_key:
        logger.error("SECRET_KEY config dosyasında belirtilmemiş.")
        raise ValueError("SECRET_KEY config dosyasında belirtilmelidir.")
    
    # Veritabanı yapılandırması
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///summaries.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum 16MB dosya yüklemesi
    
    db = SQLAlchemy(app)
    
    ALLOWED_EXTENSIONS = {'txt', 'pdf'}

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    # PDF'den metin çıkarma fonksiyonu
    def extract_text_from_pdf(filepath):
        text = ""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + " "
            logger.info("PDF'den metin başarıyla çıkarıldı.")
        except Exception as e:
            flash(f"PDF dosyasından metin çıkarılırken hata oluştu: {e}")
            logger.error(f"PDF dosyasından metin çıkarılırken hata oluştu: {e}")
        return text

    class Summary(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        title = db.Column(db.String(150), nullable=False)
        introduction = db.Column(db.Text, nullable=False)
        development = db.Column(db.Text, nullable=False)
        conclusion = db.Column(db.Text, nullable=False)
        timestamp = db.Column(db.DateTime(timezone=True), server_default=func.now())
        model_name = db.Column(db.String(50), nullable=False)
        img_intro = db.Column(db.String(200), nullable=True)
        img_development = db.Column(db.String(200), nullable=True)
        img_conclusion = db.Column(db.String(200), nullable=True)

        def __repr__(self):
            return f'<Summary {self.title}>'

    
    def save_summary_to_file(summary, filename):
        summaries_folder = 'summaries'
        os.makedirs(summaries_folder, exist_ok=True)

        file_base_name = os.path.splitext(filename)[0]
        utc_time = summary.timestamp
        utc_timestamp_filename = utc_time.strftime("%Y%m%d%H%M%S")
        model_name_safe = summary.model_name.replace("/", "_").replace("-", "_")
        
        # Dosya adı model adını içeriyor
        summary_filename = f"{file_base_name}_{model_name_safe}_{utc_timestamp_filename}.txt"
        summary_path = os.path.join(summaries_folder, summary_filename)

        try:
            with open(summary_path, 'w', encoding='utf-8') as file:
                file.write(f"Giriş:\n{summary.introduction}\n\n")
                file.write(f"Gelişme:\n{summary.development}\n\n")
                file.write(f"Sonuç:\n{summary.conclusion}\n\n")
            
            logger.info(f"Özet '{summary_filename}' dosyasına kaydedildi.")
        except Exception as e:
            logger.error(f"Özet dosyası kaydedilirken hata oluştu: {e}")


    with app.app_context():
        db.create_all()

    SUMMARY_MODEL_NAME = keys.get('SUMMARY_MODEL_NAME')
    TRANSLATION_MODEL_NAME = keys.get('TRANSLATION_MODEL_NAME')
    IMAGE_MODEL_ID = keys.get('IMAGE_MODEL_ID')

    missing_keys = []
    if not SUMMARY_MODEL_NAME:
        missing_keys.append('SUMMARY_MODEL_NAME')
    if not TRANSLATION_MODEL_NAME:
        missing_keys.append('TRANSLATION_MODEL_NAME')
    if not IMAGE_MODEL_ID:
        missing_keys.append('IMAGE_MODEL_ID')

    if missing_keys:
        logger.error(f"Config dosyasında eksik anahtarlar var: {', '.join(missing_keys)}")
        raise ValueError(f"Config dosyasında eksik anahtarlar var: {', '.join(missing_keys)}")

    logger.info(f"Özetleme modeli yüklendi: {SUMMARY_MODEL_NAME}")
    logger.info(f"Çeviri modeli yüklendi: {TRANSLATION_MODEL_NAME}")
    logger.info(f"Görsel oluşturma modeli yüklendi: {IMAGE_MODEL_ID}")

    # Modelleri yükleyelim
    summarizer_pipeline, tokenizer, max_input_length = get_summarizer_pipeline(SUMMARY_MODEL_NAME, device, logger)
    if not summarizer_pipeline:
        logger.error("Özetleme pipeline'ı yüklenemedi. Uygulama başlatılamıyor.")
        raise RuntimeError("Özetleme pipeline'ı yüklenemedi.")

    translation_model, translation_tokenizer = get_translation_models(TRANSLATION_MODEL_NAME)
    if not translation_model or not translation_tokenizer:
        logger.error("Çeviri modeli veya tokenizer yüklenemedi. Uygulama başlatılamıyor.")
        raise RuntimeError("Çeviri modeli veya tokenizer yüklenemedi.")

    pipe = get_image_pipeline(IMAGE_MODEL_ID)
    if not pipe:
        logger.error("Görsel oluşturma pipeline'ı yüklenemedi. Uygulama başlatılamıyor.")
        raise RuntimeError("Görsel oluşturma pipeline'ı yüklenemedi.")

    @app.route('/', methods=['GET', 'POST'])
    def index():
        if request.method == 'POST':
            user_text = request.form.get('user_text', '').strip()
            if not user_text:
                flash("Lütfen bir metin giriniz.")
                return redirect(url_for('index'))

            if len(user_text) > 10000:
                flash("Metin çok uzun! Maksimum 10.000 karakter girebilirsiniz.")
                return redirect(url_for('index'))

            cleaned_text = clean_text(user_text)
            introduction, development, conclusion = summarize_text(cleaned_text, summarizer_pipeline, tokenizer, max_input_length, logger)

            english_intro = translate_text(introduction, translation_model, translation_tokenizer)
            english_dev = translate_text(development, translation_model, translation_tokenizer)
            english_conc = translate_text(conclusion, translation_model, translation_tokenizer)

            file_base_name = "user_text_" + datetime.utcnow().strftime("%Y%m%d%H%M%S")

            img_intro = generate_image(english_intro, f"{file_base_name}_intro", pipe, IMAGE_MODEL_ID) or ""
            img_dev = generate_image(english_dev, f"{file_base_name}_development", pipe, IMAGE_MODEL_ID) or ""
            img_conc = generate_image(english_conc, f"{file_base_name}_conclusion", pipe, IMAGE_MODEL_ID) or ""

            new_summary = Summary(
                title="Kullanıcı Metni",
                introduction=introduction,
                development=development,
                conclusion=conclusion,
                model_name=SUMMARY_MODEL_NAME,
                img_intro=img_intro,
                img_development=img_dev,
                img_conclusion=img_conc
            )
            db.session.add(new_summary)
            db.session.commit()

            save_summary_to_file(new_summary, file_base_name + ".txt")
            flash("Özet ve görseller başarıyla oluşturuldu!")
            return redirect(url_for('summary', summary_id=new_summary.id))

        featured_summaries = Summary.query.order_by(Summary.timestamp.desc()).limit(5).all()
        logger.info("Ana sayfa görüntülendi.")
        return render_template('index.html', featured_summaries=featured_summaries)

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                uploads_folder = os.path.join(app.root_path, 'uploads')
                os.makedirs(uploads_folder, exist_ok=True)
                filepath = os.path.join(uploads_folder, filename)
                file.save(filepath)

                file_ext = filename.rsplit('.', 1)[1].lower()
                if file_ext == 'txt':
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read()
                    except Exception as e:
                        flash(f"TXT dosyasından metin okunurken hata oluştu: {e}")
                        logger.error(f"TXT dosyasından metin okunurken hata oluştu: {e}")
                        return redirect(request.url)
                elif file_ext == 'pdf':
                    text = extract_text_from_pdf(filepath)
                else:
                    flash("Desteklenmeyen dosya türü.")
                    return redirect(request.url)

                if not text.strip():
                    flash("Dosyada metin bulunamadı.")
                    return redirect(request.url)

                cleaned_text = clean_text(text)
                introduction, development, conclusion = summarize_text(cleaned_text, summarizer_pipeline, tokenizer, max_input_length, logger)

                english_intro = translate_text(introduction, translation_model, translation_tokenizer)
                english_dev = translate_text(development, translation_model, translation_tokenizer)
                english_conc = translate_text(conclusion, translation_model, translation_tokenizer)

                file_base_name = os.path.splitext(filename)[0]

                img_intro = generate_image(english_intro, f"{file_base_name}_intro", pipe, IMAGE_MODEL_ID) or ""
                img_dev = generate_image(english_dev, f"{file_base_name}_development", pipe, IMAGE_MODEL_ID) or ""
                img_conc = generate_image(english_conc, f"{file_base_name}_conclusion", pipe, IMAGE_MODEL_ID) or ""

                new_summary = Summary(
                    title=file_base_name,
                    introduction=introduction,
                    development=development,
                    conclusion=conclusion,
                    model_name=SUMMARY_MODEL_NAME,
                    img_intro=img_intro,
                    img_development=img_dev,
                    img_conclusion=img_conc
                )
                db.session.add(new_summary)
                db.session.commit()

                save_summary_to_file(new_summary, filename)
                flash("Özet ve görseller başarıyla oluşturuldu!")
                return redirect(url_for('summary', summary_id=new_summary.id))
            else:
                flash("Lütfen geçerli bir TXT veya PDF dosyası yükleyin.")
                return redirect(request.url)
        return render_template('upload.html')

    @app.route('/summary/<int:summary_id>')
    def summary(summary_id):
        summary_data = Summary.query.get_or_404(summary_id)
        logger.info(f"Özet '{summary_data.title}' görüntülendi.")
        return render_template('summary.html', summary=summary_data)

    @app.route('/search', methods=['GET', 'POST'])
    def search():
        if request.method == 'POST':
            query = request.form['query']
            results = Summary.query.filter(Summary.title.ilike(f'%{query}%')).all()
            logger.info(f"Arama yapıldı: '{query}'. {len(results)} sonuç bulundu.")
            return render_template('search.html', summaries=results, query=query)
        logger.info("Arama sayfası görüntülendi.")
        return render_template('search.html', summaries=None)

    @app.errorhandler(413)
    def request_entity_too_large(error):
        flash("Yüklenen dosya çok büyük. Maksimum 16MB.")
        logger.warning("Çok büyük bir dosya yüklendi.")
        return redirect(request.url)

    return app


if __name__ == '__main__':
    # Windows üzerinde multiprocessing ile ilgili sorunları azaltmak için TOKENIZERS_PARALLELISM kapatılabilir
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    flask_app = create_app()
    flask_app.run(debug=False)
