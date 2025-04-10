import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
@st.cache_data
def load_data():
    
    mentors = pd.read_csv("mentors.csv")
    mentors.fillna('', inplace=True)
    return mentors

mentors = load_data()

# Combine mentor features
def combine_mentor_features(row):
    return f"{row['expertise_subjects']} {row['alma_mater']} {row['experience_level']} {row['teaching_style']}"

mentors['combined'] = mentors.apply(combine_mentor_features, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
mentor_vectors = vectorizer.fit_transform(mentors['combined'])

# Streamlit UI
st.title("CLAT Mentor Recommender")

st.subheader("Tell us about yourself")

name = st.text_input("Your Name")
preferred_subjects = st.multiselect("Preferred Subjects", ["Legal Reasoning", "GK", "English", "Logical Reasoning", "Maths"])
target_colleges = st.multiselect("Target Colleges", ["NLSIU", "NALSAR", "NUJS", "NLU Delhi", "NLU Jodhpur", "NLU Patna", "NLU Bhopal"])
prep_level = st.selectbox("Your Preparation Level", ["Beginner", "Intermediate", "Advanced"])
learning_style = st.selectbox("Preferred Learning Style", ["Visual", "Reading/Writing", "Auditory", "Kinesthetic"])

if st.button("Get Mentor Recommendations"):
    if not name or not preferred_subjects or not target_colleges:
        st.warning("Please fill in all required fields.")
    else:
        # Combine input features
        combined_input = f"{', '.join(preferred_subjects)} {', '.join(target_colleges)} {prep_level} {learning_style}"
        input_vector = vectorizer.transform([combined_input])
        
        similarity_scores = cosine_similarity(input_vector, mentor_vectors).flatten()
        top_indices = similarity_scores.argsort()[-3:][::-1]
        top_mentors = mentors.iloc[top_indices]

        st.success(f"Top 3 Mentors for {name}:")
        for idx, row in top_mentors.iterrows():
            st.markdown(f"**{row['mentor_name']}** - {row['alma_mater']}  ")
            st.markdown(f"Subjects: {row['expertise_subjects']}  ")
            st.markdown(f"Experience: {row['experience_level']}, Style: {row['teaching_style']}")
            st.markdown("---")
