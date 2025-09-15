import re
import nltk
import pickle
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords
nltk.download('stopwords')

# Load preprocessed data (to get CountVectorizer vocabulary)
preprocessed_data = pd.read_csv('../data/preprocessed_data.csv')
X = preprocessed_data.drop('Sentiment', axis=1)

# Recreate CountVectorizer with same vocabulary
cv = CountVectorizer(max_features=1500, vocabulary=X.columns)

# Load trained model
with open('../data/sentiment_model.pkl', 'rb') as f:
    classifier = pickle.load(f)

ps = PorterStemmer()

def predict_sentiment(text):
    # Clean input text
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)

    # Vectorize
    review_vectorized = cv.transform([review]).toarray()

    # Predict
    prediction = classifier.predict(review_vectorized)[0]
    return "Positive 😀" if prediction == 1 else "Negative 😡"

# Example usage
if __name__ == "__main__":
    print(predict_sentiment("The food was absolutely wonderful!"))
    print(predict_sentiment("The service was terrible and slow."))
