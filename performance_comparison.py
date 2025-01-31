import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Excel dosyasını yükle
file_path = "./metrics/Model_Performance.xlsx"
df = pd.read_excel(file_path)

# Sayısal sütunlar
numeric_columns = [
    'ROUGE-1 Precision', 'ROUGE-1 Recall', 'ROUGE-1 F1',
    'ROUGE-2 Precision', 'ROUGE-2 Recall', 'ROUGE-2 F1',
    'ROUGE-L Precision', 'ROUGE-L Recall', 'ROUGE-L F1',
    'BERT Precision', 'BERT Recall', 'BERT F1'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eksik değerleri doldur
df.fillna(0, inplace=True)

# F1 değerleri
df['ROUGE-1'] = df['ROUGE-1 F1']
df['ROUGE-2'] = df['ROUGE-2 F1']
df['ROUGE-L'] = df['ROUGE-L F1']
df['BERT']    = df['BERT F1']

# Genel ortalama hesaplama (sadece F1'ler)
df['Genel Ortalama'] = df[['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT']].mean(axis=1)

# Renk paleti ayarla
palette = sns.color_palette("hsv", len(df['Model Adı'].unique()))

# Model bazında ortalamaları hesapla ve grafik oluştur
model_means = {}
for metric_name in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT', 'Genel Ortalama']:
    model_means[metric_name] = df.groupby('Model Adı')[metric_name].mean()
    means = model_means[metric_name].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.index, means.values, color=palette, edgecolor='black')
    plt.title(f'{metric_name} Model Performansı', fontsize=14)
    plt.ylabel('Değer', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Her çubuğun üstüne değerleri yazdır
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.3f}', 
                 ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'./metrics/{metric_name}_Model_Performance_Sorted.png')
    plt.show()

# Masal adına göre ortalamaları hesapla ve grafik oluştur
grouped_by_story = {}
for metric_name in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT', 'Genel Ortalama']:
    grouped_by_story[metric_name] = df.groupby('Masal Adı')[metric_name].mean()
    means = grouped_by_story[metric_name].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(means.index, means.values, color=palette, edgecolor='black')
    plt.title(f'{metric_name} Masal Performansı', fontsize=14)
    plt.ylabel('Değer', fontsize=12)
    plt.xlabel('Masal', fontsize=12)
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Her çubuğun üstüne değerleri yazdır
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.3f}', 
                 ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'./metrics/{metric_name}_Story_Performance_Sorted.png')
    plt.show()

print("Tüm grafikler başarıyla kaydedildi.")
