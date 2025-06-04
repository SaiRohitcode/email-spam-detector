import joblib
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords')

# Load the saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing function (same as before)
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return ' '.join([word for word in words if word not in stop_words])

# Predict spam or not
def predict_spam(email):
    cleaned = clean_text(email)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Test the saved model
print(predict_spam("Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's,,,"))
print(predict_spam("Hey, are we still on for dinner tonight?"))