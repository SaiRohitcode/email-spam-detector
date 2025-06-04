# Detects the spam email from entire dataset after providing the text
import tkinter as tk
from tkinter import scrolledtext, messagebox
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    stop_words = stopwords.words('english')
    return ' '.join([word for word in words if word not in stop_words])

def predict_spam():
    email_text = text_area.get("1.0", tk.END).strip()
    if not email_text:
        messagebox.showwarning("Input Error", "Please enter an email message.")
        return
    cleaned = clean_text(email_text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    result_label.config(text=f"Prediction: {result}")

# GUI setup
root = tk.Tk()
root.title("Email Spam Detector")

tk.Label(root, text="Enter Email Text:", font=("Arial", 14)).pack(pady=5)

text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=60, height=15, font=("Arial", 12))
text_area.pack(pady=5)

predict_button = tk.Button(root, text="Check Spam", command=predict_spam, font=("Arial", 14), bg="blue", fg="white")
predict_button.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ", font=("Arial", 16))
result_label.pack(pady=10)

root.mainloop()