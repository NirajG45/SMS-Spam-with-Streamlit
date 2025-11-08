import pandas as pd
import nltk
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ----------------- NLTK Setup -----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

# ----------------- Text Preprocessing -----------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ----------------- Load Datasets -----------------
print("📂 Loading datasets...")

# 1️⃣ SMS Spam dataset
sms_df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                     sep="\t", names=["label", "message"])
sms_df["label"] = sms_df["label"].map({"ham": 0, "spam": 1})
sms_df.rename(columns={"message": "text"}, inplace=True)

# 2️⃣ Email Spam dataset (Kaggle: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)
try:
    email_df = pd.read_csv("email_spam.csv")  # put your email dataset in the same folder
    email_df = email_df.rename(columns={"Category": "label", "Message": "text"})
    email_df["label"] = email_df["label"].map({"ham": 0, "spam": 1})
except Exception as e:
    print(f"⚠️ Email dataset not found. Training only on SMS dataset. ({e})")
    email_df = pd.DataFrame(columns=["label", "text"])

# 3️⃣ Combine both datasets
combined_df = pd.concat([sms_df, email_df], ignore_index=True)
print(f"✅ Combined dataset loaded: {combined_df.shape[0]} samples")

# ----------------- Preprocessing -----------------
print("🔄 Preprocessing text...")
combined_df["transformed_text"] = combined_df["text"].apply(transform_text)

# ----------------- Vectorization -----------------
print("⚙️ Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(combined_df["transformed_text"]).toarray()
y = combined_df["label"]

# ----------------- Split & Train -----------------
print("🧠 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ExtraTreesClassifier()
model.fit(X_train, y_train)

# ----------------- Evaluate -----------------
acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# ----------------- Save Model -----------------
pickle.dump(tfidf, open("vectorizer.pkl", "wb"))
pickle.dump(model, open("model.pkl", "wb"))

print("💾 Model and Vectorizer saved successfully!")
