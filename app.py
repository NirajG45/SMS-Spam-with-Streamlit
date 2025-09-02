import nltk
import streamlit as st
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sqlite3
# from sqlalchemy import create_engine

# ----------------- NLTK Setup -----------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
ps = PorterStemmer()

# ----------------- Preprocessing Function -----------------
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

# ----------------- Load Model -----------------
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Universal Spam Detector", page_icon=":shield:", layout="wide")

st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Dataset", "Database", "About", "Contact"])

# ----------------- HOME -----------------
if menu == "Home":
    st.title("📩 SMS Spam Detection Model")
    st.write("*Made with ❤️ by Niraj Kumar*")

    input_sms = st.text_area("✍️ Enter your SMS message:")

    if st.button('Predict'):
        if input_sms.strip():
            transformed_sms = transform_text(input_sms)
            vector_input = vectorizer.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            if result == 1:
                st.error("🚨 This message is classified as **Spam**")
            else:
                st.success("✅ This message is classified as **Not Spam (Ham)**")
        else:
            st.warning("⚠️ Please enter a valid SMS message.")

# ----------------- DATASET -----------------
elif menu == "Dataset":
    st.title("📊 Dataset Spam Analysis")
    uploaded_file = st.file_uploader("📂 Upload a file (CSV / Excel / JSON)", type=["csv", "xlsx", "json"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, encoding="utf-8", errors="ignore")
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_json(uploaded_file)

            st.write("✅ File uploaded successfully!")
            st.dataframe(df.head())

            text_col = st.selectbox("🔎 Select the column containing text messages:", df.columns)

            if st.button("Run Spam Detection"):
                df["transformed"] = df[text_col].astype(str).apply(transform_text)
                df["prediction"] = model.predict(vectorizer.transform(df["transformed"]))
                df["prediction_label"] = df["prediction"].map({0: "Ham", 1: "Spam"})
                st.success("🎯 Prediction Completed!")
                st.dataframe(df[[text_col, "prediction_label"]])

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download Results as CSV", data=csv, file_name="spam_detection_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"❌ Error while processing file: {e}")

# ----------------- DATABASE -----------------
elif menu == "Database":
    st.title("🗄️ Connect to Database")

    db_type = st.selectbox("Select Database Type", ["SQLite", "MySQL", "PostgreSQL"])

    if db_type == "SQLite":
        sqlite_file = st.file_uploader("📂 Upload SQLite Database (.db)", type=["db"])
        if sqlite_file:
            conn = sqlite3.connect(sqlite_file.name)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            table = st.selectbox("📋 Select Table", tables["name"])
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, conn)
            conn.close()

            st.dataframe(df.head())
            text_col = st.selectbox("🔎 Select the column containing text messages:", df.columns)

            if st.button("Run Spam Detection"):
                df["transformed"] = df[text_col].astype(str).apply(transform_text)
                df["prediction"] = model.predict(vectorizer.transform(df["transformed"]))
                df["prediction_label"] = df["prediction"].map({0: "Ham", 1: "Spam"})
                st.success("🎯 Prediction Completed!")
                st.dataframe(df[[text_col, "prediction_label"]])

    else:
        st.info("⚠️ MySQL/PostgreSQL support coming soon. Use SQLite for now.")

# ----------------- ABOUT -----------------
elif menu == "About":
    st.title("ℹ️ About SMS Spam Detector")
    st.write("""
    This project is a **Machine Learning application** built with Streamlit.

    ### 🔧 Features:
    - Predict spam for single SMS messages
    - Upload datasets (CSV, Excel, JSON) and get bulk predictions
    - Connect with SQLite databases
    - Export results as CSV

    ### 📚 Model Info:
    - Trained on **SMS Spam Collection Dataset**
    - Accuracy ~94%
    - Preprocessing includes:
      - Lowercasing
      - Tokenization
      - Stopword Removal
      - Stemming
    """)

    st.success("🚀 Built for ML Learning & Practical Experience")

    st.markdown("""
    🔗 GitHub Repository: [Click Here](https://github.com/nirajkumar/spam-detector)  
    📊 Dataset Source: [UCI SMS Spam Collection](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
    """)

# ----------------- CONTACT -----------------
elif menu == "Contact":
    st.title("📬 Contact Us")
    st.write("Have feedback or queries? Fill the form below.")

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")

        submitted = st.form_submit_button("Send")
        if submitted:
            if name and email and message:
                with open("contacts.csv", "a", encoding="utf-8") as f:
                    f.write(f"{name},{email},{message}\n")
                st.success("✅ Message sent successfully!")
            else:
                st.error("⚠️ Please fill all fields.")
