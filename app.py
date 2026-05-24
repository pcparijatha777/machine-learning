import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
movies = movies.fillna('')
movies["combined"] = movies["genres"] + " " + movies["overview"]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(movies["combined"])
similarity = cosine_similarity(vectors)

def recommend(movie_name):
    index = movies[movies['title'] == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return [movies.iloc[i[0]].title for i in sorted_scores[1:6]]

# UI
st.title("🎬 Movie Recommendation System")

movie_name = st.text_input("Enter movie name")

if st.button("Recommend"):
    try:
        results = recommend(movie_name)
        for movie in results:
            st.write(movie)
    except:
        st.write("Movie not found")