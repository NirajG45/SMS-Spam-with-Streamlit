import nltk
import streamlit as st
import pickle
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import sqlite3
import imaplib
import email
from email.header import decode_header

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

st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Dataset", "Database", "Email Scanner", "About", "Contact"])

# ----------------- HOME -----------------
if menu == "Home":
    st.title("📩 SMS / Text Spam Detection")
    st.write("*Made with ❤️ by Niraj Kumar*")

    input_sms = st.text_area("Enter your SMS message or Email content:")

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
            st.warning("Please enter a valid text message.")

# ----------------- DATASET -----------------
elif menu == "Dataset":
    st.title("📊 Dataset Spam Analysis")
    uploaded_file = st.file_uploader("Upload a file (CSV / Excel / JSON)", type=["csv", "xlsx", "json"])

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

            text_col = st.selectbox("Select the column containing text messages:", df.columns)

            if st.button("Run Spam Detection"):
                df["transformed"] = df[text_col].astype(str).apply(transform_text)
                df["prediction"] = model.predict(vectorizer.transform(df["transformed"]))
                df["prediction_label"] = df["prediction"].map({0: "Ham", 1: "Spam"})
                st.success("✅ Prediction Completed!")
                st.dataframe(df[[text_col, "prediction_label"]])

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Results as CSV", data=csv, file_name="spam_detection_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error while processing file: {e}")

# ----------------- DATABASE -----------------
elif menu == "Database":
    st.title("🗄️ Connect to Database")

    db_type = st.selectbox("Select Database Type", ["SQLite", "MySQL", "PostgreSQL"])

    if db_type == "SQLite":
        sqlite_file = st.file_uploader("Upload SQLite Database (.db)", type=["db"])
        if sqlite_file:
            conn = sqlite3.connect(sqlite_file.name)
            tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
            table = st.selectbox("Select Table", tables["name"])
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, conn)
            conn.close()

            st.dataframe(df.head())
            text_col = st.selectbox("Select the column containing text messages:", df.columns)

            if st.button("Run Spam Detection"):
                df["transformed"] = df[text_col].astype(str).apply(transform_text)
                df["prediction"] = model.predict(vectorizer.transform(df["transformed"]))
                df["prediction_label"] = df["prediction"].map({0: "Ham", 1: "Spam"})
                st.success("✅ Prediction Completed!")
                st.dataframe(df[[text_col, "prediction_label"]])

    else:
        st.info("MySQL/PostgreSQL support coming soon. Use SQLite for now.")

# ----------------- EMAIL SCANNER -----------------
elif menu == "Email Scanner":
    st.title("📧 Gmail Spam Email Detection")

    st.write("Fetch your Gmail messages and detect if they are spam or not.")
    st.info("🔐 Note: Enable 'App Passwords' in Gmail before using this feature.")

    with st.form("gmail_form"):
        gmail_user = st.text_input("Enter your Gmail address")
        gmail_pass = st.text_input("Enter your Gmail App Password", type="password")
        num_mails = st.number_input("How many recent emails to check?", min_value=1, max_value=50, value=5)
        submitted = st.form_submit_button("Fetch and Analyze")

    if submitted:
        try:
            imap = imaplib.IMAP4_SSL("imap.gmail.com")
            imap.login(gmail_user, gmail_pass)
            imap.select("inbox")
            status, messages = imap.search(None, "ALL")
            email_ids = messages[0].split()[-num_mails:]

            email_data = []
            for eid in email_ids:
                status, msg_data = imap.fetch(eid, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8", errors="ignore")

                # Extract body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body += part.get_payload(decode=True).decode("utf-8", errors="ignore")
                else:
                    body = msg.get_payload(decode=True).decode("utf-8", errors="ignore")

                text_to_check = subject + " " + body
                transformed = transform_text(text_to_check)
                pred = model.predict(vectorizer.transform([transformed]))[0]
                label = "Spam 🚨" if pred == 1 else "Not Spam ✅"
                email_data.append({"Subject": subject, "Result": label})

            imap.close()
            imap.logout()

            st.success("✅ Email analysis completed!")
            st.dataframe(pd.DataFrame(email_data))

        except Exception as e:
            st.error(f"⚠️ Error fetching emails: {e}")

# ----------------- ABOUT -----------------
elif menu == "About":
    st.title("ℹ️ About Spam Detector")
    st.write("""
    This app detects **Spam** in both SMS and Email messages.

    ### Features:
    - Predict single SMS or Email content
    - Bulk file upload (CSV, Excel, JSON)
    - SQLite database integration
    - Gmail Email Spam detection
    """)

    st.success("Built with ❤️ for learning and real-world ML practice!")

# ----------------- CONTACT -----------------
elif menu == "Contact":
    st.title("📬 Contact Us")
    st.write("Have feedback or queries? Fill the form below.")

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email_addr = st.text_input("Your Email")
        message = st.text_area("Your Message")

        submitted = st.form_submit_button("Send")
        if submitted:
            if name and email_addr and message:
                with open("contacts.csv", "a", encoding="utf-8") as f:
                    f.write(f"{name},{email_addr},{message}\n")
                st.success("Message sent successfully!")
            else:
                st.error("Please fill all fields.")
