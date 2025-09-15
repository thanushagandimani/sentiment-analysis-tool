📊 Sentiment Analysis Tool

A simple end-to-end Sentiment Analysis Project built with Python, NLTK, and Scikit-learn.
This tool analyzes text reviews (like restaurant reviews) and predicts whether they are Positive 😀 or Negative 😡.


---

📂 Project Structure

sentiment-analysis-tool/
│
├── data/
│   └── Restaurant_Reviews.tsv       # Dataset (input file)
│   └── preprocessed_data.csv        # Preprocessed dataset (created later)
│   └── sentiment_model.pkl          # Trained model (created later)
│
├── src/
│   ├── preprocess.py                # Data cleaning & preprocessing
│   ├── train_model.py               # Model training & saving
│   └── predict.py                   # Sentiment prediction
│
└── requirements.txt                 # Dependencies


---

⚙️ Installation

1. Clone the repository:

git clone https://github.com/your-username/sentiment-analysis-tool.git
cd sentiment-analysis-tool


2. Install dependencies:

pip install -r requirements.txt


3. Dataset:

Place your dataset inside data/ folder.

Example dataset: Restaurant_Reviews.tsv with columns:

Review → text review

Liked → sentiment label (1 = Positive, 0 = Negative)






---

🚀 Usage

1️⃣ Preprocess Data

Clean and prepare the dataset:

python src/preprocess.py

✔️ Creates data/preprocessed_data.csv


---

2️⃣ Train Model

Train the Naive Bayes classifier:

python src/train_model.py

✔️ Shows model accuracy
✔️ Saves model to data/sentiment_model.pkl


---

3️⃣ Predict Sentiment

Predict sentiment for new reviews:

python src/predict.py

Example output:

Positive 😀
Negative 😡


---

🛠 Features

Text preprocessing (stopword removal, stemming, cleaning)

Bag of Words (CountVectorizer) for feature extraction

Naive Bayes Classifier for training

Easy prediction on new inputs



---

📊 Example (Code Usage)

from src.predict import predict_sentiment

print(predict_sentiment("The food was absolutely wonderful!"))
# Output: Positive 😀

print(predict_sentiment("The service was terrible and slow."))
# Output: Negative 😡


---

📌 Requirements

Python 3.x

Libraries:

nltk

pandas

scikit-learn



Install via:

pip install -r requirements.txt


---

📖 License

This project is open-source and available under the MIT License.
