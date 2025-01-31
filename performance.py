import os
import difflib
from rouge import Rouge
from bert_score import score
import pandas as pd
from tqdm import tqdm

def calculate_rouge_scores(model_summary, reference_summary):
    rouge = Rouge()
    scores = rouge.get_scores(model_summary, reference_summary, avg=True)
    return {
        "ROUGE-1": scores["rouge-1"],
        "ROUGE-2": scores["rouge-2"],
        "ROUGE-L": scores["rouge-l"],
    }

def calculate_bertscore(model_summary, reference_summary):
    try:
        model_summaries = [model_summary]
        reference_summaries = [reference_summary]
        precisions, recalls, f1s = score(
            model_summaries, reference_summaries, lang="tr", model_type="xlm-roberta-base"
        )
        return {
            "Precision": round(precisions.mean().item(), 4),
            "Recall": round(recalls.mean().item(), 4),
            "F1": round(f1s.mean().item(), 4)
        }
    except Exception as e:
        print(f"BERTScore hata: {e}")
        return {"Precision": 0, "Recall": 0, "F1": 0}

def find_closest_match(file_name, folder):
    files = os.listdir(folder)
    matches = difflib.get_close_matches(file_name, files, n=1, cutoff=0.4)
    return matches[0] if matches else None

def process_files(model_folder, gpt_folder, cos_folder, output_dir="metrics"):
    os.makedirs(output_dir, exist_ok=True)
    all_data = []

    # Model klasöründeki dosyalar için ilerleme çubuğu
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".txt")]
    with tqdm(total=len(model_files), desc="Dosyalar İşleniyor", unit="dosya") as pbar:
        for model_file in model_files:
            model_path = os.path.join(model_folder, model_file)

            gpt_match = find_closest_match(model_file, gpt_folder)
            cos_match = find_closest_match(model_file, cos_folder)

            if not gpt_match or not cos_match:
                print(f"Eşleşen dosyalar bulunamadı: {model_file}")
                pbar.update(1)
                continue

            gpt_path = os.path.join(gpt_folder, gpt_match)
            cos_path = os.path.join(cos_folder, cos_match)

            with open(model_path, 'r', encoding='utf-8') as mf, \
                 open(gpt_path, 'r', encoding='utf-8') as gf, \
                 open(cos_path, 'r', encoding='utf-8') as cf:
                model_summary = mf.read()
                gpt_summary = gf.read()
                cos_summary = cf.read()

            # ROUGE ve BERTScore hesaplamaları
            gpt_rouge_scores = calculate_rouge_scores(model_summary, gpt_summary)
            gpt_bert_scores = calculate_bertscore(model_summary, gpt_summary)

            cos_rouge_scores = calculate_rouge_scores(model_summary, cos_summary)
            cos_bert_scores = calculate_bertscore(model_summary, cos_summary)

            # Verileri kaydet
            for reference, rouge_scores, bert_scores in [
                ("GPT", gpt_rouge_scores, gpt_bert_scores),
                ("Cosine", cos_rouge_scores, cos_bert_scores)
            ]:
                all_data.append({
                    "Dosya Adı": model_file,  # Dosya adını kaydediyoruz
                    "Reference": reference,
                    "ROUGE-1 Precision": round(rouge_scores["ROUGE-1"]["p"], 4),
                    "ROUGE-1 Recall": round(rouge_scores["ROUGE-1"]["r"], 4),
                    "ROUGE-1 F1": round(rouge_scores["ROUGE-1"]["f"], 4),
                    "ROUGE-2 Precision": round(rouge_scores["ROUGE-2"]["p"], 4),
                    "ROUGE-2 Recall": round(rouge_scores["ROUGE-2"]["r"], 4),
                    "ROUGE-2 F1": round(rouge_scores["ROUGE-2"]["f"], 4),
                    "ROUGE-L Precision": round(rouge_scores["ROUGE-L"]["p"], 4),
                    "ROUGE-L Recall": round(rouge_scores["ROUGE-L"]["r"], 4),
                    "ROUGE-L F1": round(rouge_scores["ROUGE-L"]["f"], 4),
                    "BERT Precision": bert_scores["Precision"],
                    "BERT Recall": bert_scores["Recall"],
                    "BERT F1": bert_scores["F1"]
                })

            # İlerleme çubuğunu güncelle
            pbar.update(1)

    # Verileri Excel'e kaydet
    df = pd.DataFrame(all_data)
    excel_path = os.path.join(output_dir, "Model_Performance.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"Performans tablosu '{excel_path}' olarak kaydedildi.")

# Klasör yollarını belirtin
model_folder = "summaries"
gpt_folder = "gpt_summaries"
cos_folder = "cos_sim_summaries"

# Çalıştır
process_files(model_folder, gpt_folder, cos_folder)
