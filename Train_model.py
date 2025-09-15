import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle

# Load preprocessed data
preprocessed_data = pd.read_csv('../data/preprocessed_data.csv')

# Split features and labels
X = preprocessed_data.drop('Sentiment', axis=1)
y = preprocessed_data['Sentiment']

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {accuracy:.2f}")

# Save model for later use
with open('../data/sentiment_model.pkl', 'wb') as f:
    pickle.dump(classifier, f)

print("✅ Model saved to data/sentiment_model.pkl")
