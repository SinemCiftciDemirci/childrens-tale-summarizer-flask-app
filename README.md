# childrens-tale-summarizer
# Overview
Using the summarization models available on the Hugging Face Platform with Turkish language support, Turkish children's tales were summarized and the performance of these models was compared. As a result of this performance analysis, the parameters of the three most successful models were adjusted specifically for the model. For this, the most suitable tale for summarization was selected from among the tales used and the parameters were adjusted according to the success here. Tales were divided into three sections as introduction-development-conclusion, and visuals were produced with a separate summary for each section and a visual production model for each summary.

# Features
📜 Tale Summarization: Summarizes Turkish children's tales using AI-powered NLP models.
📊 Model Performance Comparison: Evaluates different summarization models using ROUGE and BERTScore metrics.
🎨 AI-Powered Visual Generation: Generates illustrations for each section of the story using text-to-image models.
📝 Text and PDF Support: Users can upload TXT or PDF files to be summarized.
🌍 English Translation: Summarized texts are translated into English for visualization purposes.

# 📂 Project Structure

├── app.py                     # Flask web application
├── config.txt                  # API keys and model configurations
├── config_loader.py            # Loads configuration files
├── image_generator.py          # AI-based image generation
├── requirements.txt            # Required dependencies
├── summarizer.py               # Main summarization logic
├── templates/                  # HTML templates for Flask


# 🛠 Installation & Setup
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Web Application
python app.py

The app will be accessible at local.

# 📑 Usage Guide
1. Upload a Story
Upload a TXT or PDF file containing a children's tale.

3. Summarization & Translation
The story is automatically summarized, divided into three sections (Introduction, Development, Conclusion), and translated into English.

4. Image Generation
Illustrations are generated for each section using a text-to-image model.

5. Performance Analysis
The summarization quality is measured using ROUGE and BERTScore.

6. Results Storage
Summaries and images are saved for further evaluation.

# ⚙️ Models Used
Summarization Model: Turkish-NLP/t5-efficient-small-MLSUM-TR-fine-tuned
Translation Model: Helsinki-NLP/opus-mt-tr-en
Image Generation Model: stable-diffusion-v1-5/stable-diffusion-v1-5


# 📊 Performance Evaluation
The project includes a performance assessment of different summarization models using:

ROUGE-1, ROUGE-2, ROUGE-L (Precision, Recall, F1-score)
BERTScore (Evaluation against reference summaries)
To evaluate model performance:
python performance.py

Comparison metrics are stored in the metrics/ directory.


# 🔑 Configuration
Add your API keys and model configurations in the config.txt file:

SECRET_KEY= "your_secret_key"
HUGGINGFACE_API_KEY= "your_huggingface_api_key"
OPENAI_API_KEY= "your_openai_api_key"
SUMMARY_MODEL_NAME= "Turkish-NLP/t5-efficient-small-MLSUM-TR-fine-tuned"
TRANSLATION_MODEL_NAME= "Helsinki-NLP/opus-mt-tr-en"
IMAGE_MODEL_ID= "stable-diffusion-v1-5/stable-diffusion-v1-5"


# 📬 Contribution
Contributions are welcome! Feel free to:

Fork the repository
Submit a pull request
Report issues and suggest improvements
