# 📧 Spam Email Detector – AI Internship Project

This is a simple spam detection project using machine learning and natural language processing. It is built using Python, scikit-learn, and nltk. The project trains a model on a real dataset, predicts spam/ham messages, and includes both CLI and GUI-based interfaces.

---

## 📁 Project Files Overview

| File Name              | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| spam_detector.py     | ✅ Trains model & predicts spam/ham for *2 example messages*              |
| spam_detect_all.py   | ✅ Predicts spam/ham for *all messages in the dataset*                    |
| spam_gui.py           | ✅ Simple GUI to test custom message input                                  |
| spam.csv             | 📄 Dataset file (UCI SMS Spam Collection)                                   |
| spam_model.pkl       | 💾 Trained model saved for reuse                                            |
| tfidf_vectorizer.pkl | 💾 Saved TF-IDF vectorizer used to convert text into numerical form         | --> for gui
| vectorizer.pkl | 💾 Saved TF-IDF vectorizer used to convert text into numerical form         | --> for remaining 2 programs
| README.md            | 📘 Project documentation (this file)                                        |

---

 🔍 1. spam_detector.py – Predicts on Two Example Messages

This script:

- Loads and preprocesses the dataset
- Trains a Naive Bayes model
- Evaluates model accuracy
- Predicts spam status for *2 sample messages*

 ✅ To Run:
bash
python spam_detector.py


 🔍 2. spam_detect_all.py – Predicts on entire dataset

This script:

- Loads and preprocesses the dataset
- Trains a Naive Bayes model
- Evaluates model accuracy
- Predicts spam status for entire dataset

 ✅ To Run:
bash
python spam_detect_all.py

🔍 3. spam_gui.py – Predicts on entire dataset upon giving the input

This script:

- Loads and preprocesses the dataset
- Trains a Naive Bayes model
- Evaluates model accuracy
- Predicts spam status for entire dataset by providing the email required to be checked

 ✅ To Run:
bash
python spam_gui.py 
