# Detects the spam email from entire dataset
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Download stopwords if not already
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocess text
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return ' '.join([word for word in words if word not in stop_words])

df['cleaned'] = df['message'].apply(clean_text)

# Convert text to TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])

# Labels: ham=0, spam=1
y = df['label'].map({'ham': 0, 'spam': 1})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to test on new messages
def predict_spam(email):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Test examples (optional)
print(predict_spam("Congratulations! You've won a free ticket to Bahamas! Click here."))
print(predict_spam("Hey, are we still on for dinner tonight?"))

# === NEW: Batch prediction for all messages in the dataset ===
print("\nPredictions on entire dataset:\n")
for msg in df['message']:
    print(f"Message: {msg}")
    print("Prediction:", predict_spam(msg))
    print("---")

    # Save the trained model
joblib.dump(model, 'spam_model.pkl')

# Optional: Save the vectorizer too!
joblib.dump(vectorizer, 'vectorizer.pkl')
