import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="🚀",
    layout="centered"
)

# ── Load dataset ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("raw_skills.csv")
    return df

df = load_data()

# ── TF-IDF Vectorizer ────────────────────────────────────────
@st.cache_resource
def build_vectorizer(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_vectorizer(df)

# ── Recommendation Function ───────────────────────────────────
def recommend(user_skills, top_n=3):
    user_input = " ".join(user_skills * 3)
    user_vector = vectorizer.transform([user_input])
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_n]
    results = []
    for i in top_indices:
        results.append({
            "Job Role": df.iloc[i]["job_role"],
            "Match Score": f"{scores[i]*100:.1f}%",
            "Key Skills": df.iloc[i]["skills"].replace("_", " ")
        })
    return results

# ── UI ────────────────────────────────────────────────────────
st.title("🚀 Tech Stack Recommender")
st.markdown("#### *Powered by TF-IDF + Cosine Similarity*")
st.markdown("---")

st.markdown("### 👤 Enter Your Skills Below")
st.markdown("Type **at least 3 skills** you know (e.g. Python, SQL, Machine_Learning)")

col1, col2, col3 = st.columns(3)
with col1:
    skill1 = st.text_input("Skill 1", placeholder="e.g. Python")
with col2:
    skill2 = st.text_input("Skill 2", placeholder="e.g. SQL")
with col3:
    skill3 = st.text_input("Skill 3", placeholder="e.g. Machine_Learning")

skill4 = st.text_input("Skill 4 (optional)", placeholder="e.g. TensorFlow")
skill5 = st.text_input("Skill 5 (optional)", placeholder="e.g. Docker")

st.markdown("---")

if st.button("🔍 Find My Best Career Matches", use_container_width=True):
    user_skills = [s.strip() for s in [skill1, skill2, skill3, skill4, skill5] if s.strip()]
    
    if len(user_skills) < 3:
        st.warning("⚠️ Please enter at least 3 skills to get accurate recommendations.")
    else:
        st.markdown("### 🎯 Your Top 3 Career Path Matches")
        results = recommend(user_skills, top_n=3)
        
        medals = ["🥇", "🥈", "🥉"]
        for idx, result in enumerate(results):
            with st.container():
                st.markdown(f"## {medals[idx]} {result['Job Role']}")
                st.success(f"**Match Score: {result['Match Score']}**")
                st.markdown(f"**Key Skills for this role:** {result['Key Skills']}")
                st.markdown("---")

st.markdown(
    "<br><center><small>Built with ❤️ using Python · TF-IDF · Cosine Similarity · Streamlit</small></center>",
    unsafe_allow_html=True
)