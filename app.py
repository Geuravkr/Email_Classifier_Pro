
import streamlit as st
import pandas as pd
import string
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Spam AI Pro",
    page_icon="📧",
    layout="wide"
)

# ---------------------------------------------------
# PREMIUM GLASSMORPHISM CSS
# ---------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
}
.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go To",
    ["🏠 Home", "📊 Analytics", "📚 Data Structures", "👨‍💻 About"]
)

# ---------------------------------------------------
# DATA STRUCTURE FUNCTIONS
# ---------------------------------------------------
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()  # LIST

def build_frequency_dict(messages):
    freq_dict = {}  # HASHMAP
    for message in messages:
        words = preprocess_text(message)
        for word in words:
            freq_dict[word] = freq_dict.get(word, 0) + 1
    return freq_dict

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    return df

df = load_data()

# ---------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------------------------------------------
# VOCABULARY USING SET
# ---------------------------------------------------
vocab_set = set()
for msg in df['message']:
    vocab_set.update(preprocess_text(msg))

# ===================================================
# 🏠 HOME PAGE
# ===================================================
if page == "🏠 Home":

    st.markdown("<h1 style='text-align:center;'>📧 Spam AI Pro</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:#ccc;'>AI-Powered Email Classification System</h4>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Accuracy", f"{accuracy:.2f}")
    col2.metric("📚 Vocabulary Size", len(vocab_set))
    col3.metric("📩 Total Emails", len(df))

    st.markdown("---")
    st.subheader("✍️ Enter Email Content")

    user_input = st.text_area("", height=200, placeholder="Paste email content here...")

    if st.button("🚀 Analyze Email"):
        if user_input.strip() != "":
            with st.spinner("Analyzing with AI Engine..."):
                time.sleep(1.5)

            input_vector = vectorizer.transform([user_input])
            prediction = model.predict(input_vector)[0]
            probability = model.predict_proba(input_vector)
            confidence = max(probability[0]) * 100

            st.markdown("## 🔎 Prediction Result")

            if prediction == "spam":
                st.error(f"🚨 SPAM ({confidence:.2f}% Confidence)")
            else:
                st.success(f"✅ NOT SPAM ({confidence:.2f}% Confidence)")

            # Word Frequency
            words = preprocess_text(user_input)
            freq = Counter(words)

            with st.expander("📌 View Word Frequency (Dictionary Implementation)"):
                st.write(dict(freq))

# ===================================================
elif page == "📊 Analytics":

    st.title("📊 AI Analytics Dashboard")

    # ---------- STYLE SETTINGS ----------
    plt.style.use("dark_background")

    # ==============================
    # 1️⃣ Spam vs Ham Donut Chart
    # ==============================
    st.subheader("📩 Email Distribution")

    label_counts = df['label'].value_counts()

    fig1, ax1 = plt.subplots(figsize=(6,6))

    colors = ["#00c6ff", "#ff4b5c"]

    wedges, texts, autotexts = ax1.pie(
        label_counts,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4)  # Makes it donut style
    )

    ax1.set_title("Spam vs Ham Distribution", fontsize=14, weight="bold")
    st.pyplot(fig1)


    # ==============================
    # 2️⃣ Top 10 Words Bar Chart
    # ==============================
    st.subheader("🔥 Top 10 Frequent Words")

    freq_dict = build_frequency_dict(df['message'])
    top_words = Counter(freq_dict).most_common(10)

    words = [w[0] for w in top_words]
    counts = [w[1] for w in top_words]

    fig2, ax2 = plt.subplots(figsize=(8,5))

    bars = ax2.bar(words, counts, color="#00c6ff", alpha=0.8)

    ax2.set_title("Most Frequent Words", fontsize=14, weight="bold")
    ax2.set_xlabel("Words")
    ax2.set_ylabel("Frequency")

    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)

    st.pyplot(fig2)


# ===================================================
# 📚 DATA STRUCTURES PAGE
# ===================================================
elif page == "📚 Data Structures":

    st.title("📚 Data Structures Used")

    st.markdown("""
    ### 🔹 1. List
    Used for tokenizing words.

    ### 🔹 2. Set
    Used to create unique vocabulary.
    Time Complexity: O(1) average insertion.

    ### 🔹 3. Dictionary (HashMap)
    Used to store word frequencies.

    ### 🔹 4. Counter
    Used to compute top frequent words efficiently.

    ### 🔹 5. Sparse Matrix
    Used internally by CountVectorizer for memory efficiency.
    """)

# ===================================================
# 👨‍💻 ABOUT PAGE
# ===================================================
elif page == "👨‍💻 About":

    st.title("👨‍💻 About This Project")

    st.markdown("""
    **Spam AI Pro** is a Data Structures + Machine Learning project.

    ✔ Uses HashMaps, Sets, Lists  
    ✔ Implements Multinomial Naive Bayes  
    ✔ Real-time classification  
    ✔ Analytics dashboard  

    Built using Streamlit.
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Built by Team Unknown | Data Structures & ML Project</center>",
    unsafe_allow_html=True
)