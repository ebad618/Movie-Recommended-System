import streamlit as st
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = "e731d777812df1b8cd76a37364b3eeaf"  # consider moving to env var

def fetch_poster(movie_id: int | str):
    """Return full TMDB poster URL for a movie_id, or None if unavailable."""
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        return None

# ---------- Load data & models ----------
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

# These must be the SAME vectorizer & vectors you trained with
cv = pickle.load(open('cv.pkl', 'rb'))          # CountVectorizer object
vectors = pickle.load(open('vectors.pkl', 'rb'))  # document-term matrix

# Try to detect which column holds TMDB IDs
ID_COL = next((c for c in ['movie_id', 'tmdb_id', 'tmdbId', 'id', 'movieId'] if c in movies.columns), None)

def recommend_by_text(query: str, top_n: int = 25):
    """Return (names, posters) for the top_n most similar movies to the text query."""
    query_vec = cv.transform([query])
    similarity_scores = cosine_similarity(query_vec, vectors).ravel()
    top_idx = similarity_scores.argsort()[-top_n:][::-1]

    names = movies.iloc[top_idx]['title'].tolist()

    posters = []
    if ID_COL is not None:
        for i in top_idx:
            movie_id = movies.iloc[i][ID_COL]
            posters.append(fetch_poster(movie_id))
    else:
        posters = [None] * len(names)

    return names, posters

# ---------- UI ----------
st.title('Movie Recommender System')
user_input = st.text_input('Enter any text (plot, vibe, keywords) to get movie recommendations:')

if st.button('Recommend'):
    text = user_input.strip()
    if not text:
        st.warning("Please enter some text to get recommendations.")
    else:
        names, posters = recommend_by_text(text, top_n=25)

        # Grid: up to 5x5 (25 items)
        cols = st.columns(5)
        for row in range(5):
            for col in range(5):
                k = row * 5 + col
                if k >= len(names):
                    break
                with cols[col]:
                    st.text(names[k])
                    if posters[k]:
                        st.image(posters[k], use_container_width=True)



        # Also list them plainly below
        st.markdown("### Top Recommendations:")
        for i, name in enumerate(names, start=1):
            st.write(f"{i}. {name}")
