import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download stopwords if not already present
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('../data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Initialize tools
ps = PorterStemmer()
corpus = []

# Preprocess each review
for review in data['Review']:
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    corpus.append(' '.join(review))

# Convert text to numerical data
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = data['Liked']

# Save preprocessed data
preprocessed_data = pd.DataFrame(X, columns=cv.get_feature_names_out())
preprocessed_data['Sentiment'] = y
preprocessed_data.to_csv('../data/preprocessed_data.csv', index=False)

print("✅ Preprocessing completed. Data saved to data/preprocessed_data.csv")
