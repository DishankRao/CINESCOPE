# app.py - CineScope (Full) - PART 1
import streamlit as st
import requests
from transformers import pipeline
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objects as go
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# -------------------------------
# TMDb API Key
# -------------------------------
TMDB_API_KEY = "a263aecaffb6bb39c96f5b2a4fbd6073"
TMDB_BASE = "https://api.themoviedb.org/3"

# -------------------------------
# Lightweight tokenizer & stopwords (no NLTK downloads)
# -------------------------------
_SIMPLE_STOPWORDS = {
    "the", "and", "a", "an", "in", "on", "for", "of", "to", "is", "it",
    "this", "that", "with", "as", "its", "was", "are", "by", "at", "be",
    "from", "has", "have", "but", "not", "or", "if", "they", "their",
    "i", "we", "you", "he", "she", "them", "his", "her", "my", "me"
}

def simple_tokenize(text):
    """Very lightweight tokenizer that avoids external NLTK downloads."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", (text or "").lower())
    return [w for w in words if w not in _SIMPLE_STOPWORDS]

# -------------------------------
# Load HuggingFace Pipelines (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def load_pipelines():
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarizer = pipeline("summarization", model="facebook/bart-base")
    return sentiment, summarizer

# Load (may take time on first run)
sentiment_pipeline, summarizer_pipeline = load_pipelines()

# -------------------------------
# TMDb API Helpers (movies / tv / anime)
# -------------------------------
def _filter_adult(items, include_adult: bool):
    """Filter out adult items unless include_adult True."""
    if include_adult:
        return items
    return [i for i in items if not i.get("adult", False)]

def search_content(query, tmdb_type="movie", include_adult=False):
    """
    Search TMDb for movies or TV shows.
    tmdb_type = 'movie' or 'tv'
    """
    url = f"{TMDB_BASE}/search/{tmdb_type}"
    params = {"api_key": TMDB_API_KEY, "query": query, "language": "en-US", "page": 1, "include_adult": str(include_adult).lower()}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []
    return _filter_adult(results, include_adult)

def get_content_details(content_id, tmdb_type="movie"):
    url = f"{TMDB_BASE}/{tmdb_type}/{content_id}"
    params = {"api_key": TMDB_API_KEY, "append_to_response": "credits", "language": "en-US"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

def get_content_reviews(content_id, tmdb_type="movie", max_reviews=20):
    url = f"{TMDB_BASE}/{tmdb_type}/{content_id}/reviews"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    texts = [item.get("content", "") for item in data.get("results", []) if item.get("content")]
    return texts[:max_reviews]

def get_recommendations(content_id, tmdb_type="movie", max_recs=8, include_adult=False):
    url = f"{TMDB_BASE}/{tmdb_type}/{content_id}/recommendations"
    params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []
    return _filter_adult(results, include_adult)[:max_recs]

def fetch_popular(tmdb_type="movie", pages=2, include_adult=False):
    items = []
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE}/{tmdb_type}/popular"
        params = {"api_key": TMDB_API_KEY, "language": "en-US", "page": page}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        items.extend(data.get("results", []) or [])
    return _filter_adult(items, include_adult)

def get_content_trailer(content_id, tmdb_type="movie"):
    url = f"{TMDB_BASE}/{tmdb_type}/{content_id}/videos"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    for v in data.get("results", []):
        if v.get("type") == "Trailer" and v.get("site") == "YouTube":
            return v.get("key")
    return None

def get_indian_movies(endpoint="now_playing", pages=1, include_adult=False):
    """
    Discover movies filtered to Indian origin.
    endpoint: 'now_playing' | 'upcoming' | 'top_rated'
    """
    movies = []
    today = datetime.date.today().strftime("%Y-%m-%d")

    for page in range(1, pages + 1):
        base_params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US",
            "with_origin_country": "IN",   # ✅ Only movies produced in India
            "include_adult": str(include_adult).lower(),
            "page": page
        }

        if endpoint == "top_rated":
            params = {**base_params, "sort_by": "vote_average.desc", "vote_count.gte": 100}
        elif endpoint == "now_playing":
            params = {**base_params, "primary_release_date.lte": today, "sort_by": "release_date.desc"}
        elif endpoint == "upcoming":
            params = {**base_params, "primary_release_date.gte": today, "sort_by": "release_date.asc"}
        else:
            params = base_params

        r = requests.get(f"{TMDB_BASE}/discover/movie", params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", []) or []

        # ✅ Extra filtering: Keep only movies in major Indian languages
        valid_langs = {"hi", "ta", "te", "ml", "kn", "bn", "mr", "pa", "gu", "or", "as"}
        filtered = [m for m in results if m.get("original_language") in valid_langs]

        # ✅ Ensure poster exists
        for m in filtered:
            if not m.get("poster_path"):
                m["poster_path"] = None

        movies.extend(filtered)

    return _filter_adult(movies, include_adult)


def get_anime_trending(pages=1, include_adult=False):
    anime = []
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE}/discover/tv"
        params = {
            "api_key": TMDB_API_KEY,
            "with_original_language": "ja",
            "with_genres": "16",  # Animation
            "sort_by": "popularity.desc",
            "page": page
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        anime.extend(data.get("results", []) or [])
    return _filter_adult(anime, include_adult)

def make_bookmyshow_url(title: str) -> str:
    """
    Generate a BookMyShow search URL for a given movie title.
    """
    safe_title = title.replace(" ", "+")
    return f"https://in.bookmyshow.com/explore/movies-{safe_title}"


# Backwards-compatible aliases
def fetch_popular_movies(pages=2, include_adult=False):
    return fetch_popular(tmdb_type="movie", pages=pages, include_adult=include_adult)

def get_movie_trailer(movie_id):
    return get_content_trailer(movie_id, tmdb_type="movie")

def get_movie_details(movie_id):
    return get_content_details(movie_id, tmdb_type="movie")

def get_movie_reviews(movie_id, max_reviews=20):
    return get_content_reviews(movie_id, tmdb_type="movie", max_reviews=max_reviews)

# -------------------------------
# AI Helpers
# -------------------------------
def analyze_sentiment_batch(reviews):
    if not reviews:
        return 0, 0
    try:
        results = sentiment_pipeline(reviews, truncation=True)
    except Exception:
        # fallback: simple heuristic if pipeline fails
        pos = sum(1 for r in reviews if "good" in (r or "").lower() or "love" in (r or "").lower())
        neg = len(reviews) - pos
        return pos, neg
    pos = sum(1 for r in results if r["label"].upper().startswith("POS"))
    neg = sum(1 for r in results if r["label"].upper().startswith("NEG"))
    return pos, neg

def summarize_reviews(reviews, max_length=120):
    if not reviews:
        return "No reviews available to summarize."
    text = " ".join(reviews[:6])
    try:
        out = summarizer_pipeline(text[:1000], max_length=max_length, min_length=20, do_sample=False)
        return out[0].get("summary_text", "Summary unavailable.")
    except Exception:
        return "Summary unavailable (model error)."

# -------------------------------
# Simple Box Office predictor (demo)
# -------------------------------
X = np.array([
    [100, 80, 120, 7.5],
    [10, 20, 90, 6.0],
    [50, 60, 110, 7.0],
    [200, 90, 150, 8.0],
    [5, 15, 85, 5.5]
])
y = np.array([500, 30, 200, 900, 10])  # simulated revenues in $M
box_office_model = LinearRegression().fit(X, y)

def predict_box_office(budget, popularity, runtime, vote_avg):
    features = np.array([[budget, popularity, runtime, vote_avg]])
    try:
        pred = box_office_model.predict(features)[0]
    except Exception:
        pred = 0
    return round(float(pred), 2)

# -------------------------------
# Analysis helpers
# -------------------------------
def analyze_person(person_id, role="Director"):
    url = f"{TMDB_BASE}/person/{person_id}/movie_credits"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    movies = data.get("crew" if role == "Director" else "cast", []) or []
    revenues, ratings = [], []
    for m in movies:
        rev = m.get("revenue", 0)
        if not rev:
            rev = m.get("popularity", 0) * 1e6
        if rev > 0:
            revenues.append(rev)
        if m.get("vote_average", 0):
            ratings.append(m["vote_average"])
    if revenues:
        avg_rev = round(np.mean(revenues) / 1e6, 2)
        avg_rat = round(np.mean(ratings), 2) if ratings else None
        return avg_rev, avg_rat, len(revenues)
    return None, None, 0

def get_genre_movies(genre_id, pages=2):
    revenues = []
    for page in range(1, pages + 1):
        url = f"{TMDB_BASE}/discover/movie"
        params = {"api_key": TMDB_API_KEY, "with_genres": genre_id, "sort_by": "popularity.desc", "page": page}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        for m in data.get("results", []) or []:
            rev = m.get("revenue", 0)
            if not rev:
                rev = m.get("popularity", 0) * 1e6
            if rev > 0:
                revenues.append(rev)
    return revenues

def get_director_trend(person_id):
    url = f"{TMDB_BASE}/person/{person_id}/movie_credits"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    movies = data.get("crew", []) or []
    trend_data = []
    for m in movies:
        if m.get("job") == "Director" and m.get("release_date"):
            try:
                year = int(m["release_date"][:4])
            except Exception:
                continue
            rev = m.get("revenue", 0)
            if not rev:
                rev = m.get("popularity", 0) * 1e6
            trend_data.append((year, rev))
    trend_data.sort(key=lambda x: x[0])
    return trend_data

def calculate_similarity(base, candidate):
    score = 0
    base_genres = {g["id"] for g in base.get("genres", [])} if base.get("genres") else set()
    cand_genres = set(candidate.get("genre_ids", []))
    score += len(base_genres & cand_genres) * 3
    if base.get("vote_average") and candidate.get("vote_average"):
        score -= abs(base["vote_average"] - candidate["vote_average"])
    return score

# -------------------------------
# Session state for navigation + toggles
# -------------------------------
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None
if "selected_content_type" not in st.session_state:
    st.session_state.selected_content_type = "movie"
if "include_adult" not in st.session_state:
    st.session_state.include_adult = False  # default OFF

# app.py - CineScope (Full) - PART 2 (REPLACEMENT)
import urllib.parse
import datetime

# -------------------------------
# Streamlit UI (Part 2)
# -------------------------------
st.set_page_config(page_title="CineScope 🎬", page_icon="🎥", layout="wide")
st.title("🎬 CineScope — AI Movie Insights")

# -------------------------------
# Helper UI session defaults
# -------------------------------
if "include_adult" not in st.session_state:
    st.session_state.include_adult = False
if "selected_movie_id" not in st.session_state:
    st.session_state.selected_movie_id = None
if "selected_content_type" not in st.session_state:
    st.session_state.selected_content_type = "movie"
if "reco_blocklist" not in st.session_state:
    st.session_state.reco_blocklist = []  # list of lowercase keywords to hide from recommendations
if "hide_blocked_recs" not in st.session_state:
    st.session_state.hide_blocked_recs = True

# -------------------------------
# Small helper utilities (used by UI)
# -------------------------------
def _filter_adult(items, include_adult: bool):
    """Return items filtered by 'adult' flag if include_adult is False."""
    if include_adult:
        return items or []
    filtered = [m for m in (items or []) if not m.get("adult")]
    return filtered

def make_bookmyshow_url(title: str, city: str = "mumbai"):
    """Return a BookMyShow search URL for Indian booking (search-based)."""
    if not title:
        return "https://in.bookmyshow.com/"
    q = urllib.parse.quote_plus(title)
    return f"https://in.bookmyshow.com/search?q={q}"

def is_blocked_by_keywords(item, keywords):
    """Return True if the title/name contains any of the keywords."""
    if not keywords:
        return False
    title = (item.get("title") or item.get("name") or "").lower()
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        if kw in title:
            return True
    return False

def fetch_popular_safe(tmdb_type="movie", pages=2, include_adult=False):
    """Wrapper around fetch_popular (from Part1). Filters adult and returns list."""
    try:
        items = fetch_popular(tmdb_type=tmdb_type, pages=pages)  # uses function from PART 1
    except Exception:
        items = []
    return _filter_adult(items, include_adult)

# -------------------------------
# Sidebar - Global Controls
# -------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    # 🔹 Content Type FIRST (only once!)
    content_choice = st.selectbox("🎞️ Content Type", ["Movie", "TV Series", "Anime"])

    # 🔹 Adult toggle
    st.session_state.include_adult = st.checkbox(
        "🔞 Include Adult Results",
        value=st.session_state.include_adult
    )

    st.markdown("---")

    # 🔹 Recommendation Filters (compact)
    st.subheader("🎯 Recommendations")
    st.session_state.hide_blocked_recs = st.checkbox(
        "Hide Blocklisted Items",
        value=st.session_state.hide_blocked_recs
    )

    new_block_input = st.text_input("Block Keywords (comma-separated)", value="")
    col_add, col_clear = st.columns([1,1])
    with col_add:
        if st.button("➕ Add", use_container_width=True):
            if new_block_input.strip():
                parts = [p.strip().lower() for p in new_block_input.split(",") if p.strip()]
                st.session_state.reco_blocklist = list(dict.fromkeys(st.session_state.reco_blocklist + parts))
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.reco_blocklist = []

    if st.session_state.reco_blocklist:
        st.caption("Blocked: " + ", ".join(st.session_state.reco_blocklist))

    st.markdown("---")

    # 🔹 Preferences (future options)
    st.subheader("🌐 Preferences")
    region_choice = st.selectbox("Preferred Region", ["Global", "India"])
    # Removed theme toggle (does not work in Streamlit dynamically)

# -------------------------------
# Back button (visible when a poster was clicked)
# -------------------------------
if st.session_state.get("selected_movie_id"):
    if st.button("⬅️ Back to Search"):
        st.session_state.selected_movie_id = None
        st.session_state.selected_content_type = "movie"

# -------------------------------
# Map content type to TMDb API type
# -------------------------------
if content_choice == "Movie":
    tmdb_type = "movie"
elif content_choice == "TV Series":
    tmdb_type = "tv"
else:  # Anime uses TV endpoint but filtered specially
    tmdb_type = "tv"

# -------------------------------
# Search box (main area)
# -------------------------------
search_query = st.text_input("🔍 Search for a Movie / Series / Anime", value="")

selected_item = None
if search_query.strip():
    try:
        results = search_content(search_query, tmdb_type=tmdb_type)  # uses function from PART 1
    except Exception:
        results = []
    results = _filter_adult(results, st.session_state.include_adult)

    if content_choice == "Anime":
        filtered = [r for r in results if r.get("original_language") == "ja" or (16 in r.get("genre_ids", []))]
        if filtered:
            results = filtered

    results = results[:6]
    if results:
        options = []
        for r in results:
            title = r.get("title") or r.get("name") or "Unknown"
            year = (r.get("release_date") or r.get("first_air_date") or "")[:4]
            options.append(f"{title} ({year})")
        selected_label = st.selectbox("Select an item:", options)
        selected_item = results[options.index(selected_label)]

# -------------------------------
# Homepage (shown if no search & no poster selected)
# -------------------------------
if not search_query.strip() and not st.session_state.get("selected_movie_id"):
    st.markdown(
        """
        <style>
        .stApp {
            background: url('https://wallpapercave.com/wp/wp6980081.jpg');
            background-size: cover;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="text-align: center; margin-top: 10vh;">
            <h1 style="font-size: 3.2em; color: #E50914;">🍿 CineScope</h1>
            <h3 style="color: #ddd;">AI Movie Insights, Reviews & Predictions</h3>
            <p style="color: #aaa;">Search any movie, series, or anime to get started!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

# -------------------------------
# Always create tabs so no NameError occurs
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["🎬 Overview", "🧠 Insights", "📊 Analytics", "💰 Predictions",
     "🎥 Recommendations", "📅 Now Playing", "🇮🇳 Indian Movies"]
)

# -------------------------------
# If a poster was clicked earlier, or a selection made, set the content_id/api_type
# -------------------------------
details = None
reviews = []
cast_list = []
director_obj = None
api_type = "movie"
content_id = None

if st.session_state.get("selected_movie_id"):
    content_id = st.session_state.selected_movie_id
    api_type = st.session_state.get("selected_content_type", "movie")
elif selected_item:
    content_id = selected_item.get("id")
    api_type = tmdb_type

if content_id:
    try:
        details = get_content_details(content_id, api_type)
    except Exception:
        details = None

    if details:
        try:
            reviews = get_content_reviews(content_id, api_type, max_reviews=20)
        except Exception:
            reviews = []
        cast_list = details.get("credits", {}).get("cast", [])[:6]
        crew_list = details.get("credits", {}).get("crew", []) or []
        director_obj = next((c for c in crew_list if c.get("job") == "Director"), None)

# -------------------------------
# Overview Tab
# -------------------------------
with tab1:
    if not details:
        st.info("Search or select an item to see its overview here.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            poster = details.get("poster_path")
            if poster:
                st.image(f"https://image.tmdb.org/t/p/w500{poster}", width=420)
            else:
                st.info("Poster not available.")
        with col2:
            title = details.get("title") or details.get("name") or "Unknown"
            year = (details.get("release_date") or details.get("first_air_date") or "")[:4]
            rating = details.get("vote_average", "N/A")
            st.subheader(f"{title} ({year}) ⭐ {rating}/10")
            genres = ", ".join([g.get("name") for g in details.get("genres", [])]) if details.get("genres") else "N/A"
            st.write(f"**Genres:** {genres}")
            st.write(f"**Overview:** {details.get('overview', 'No overview available.')}")

        st.markdown("---")
        st.subheader("🎭 Cast & Crew")
        st.write(f"**Director:** {director_obj['name'] if director_obj else 'Unknown'}")
        if cast_list:
            cast_cols = st.columns(len(cast_list))
            for col, actor in zip(cast_cols, cast_list):
                with col:
                    if actor.get("profile_path"):
                        st.image(f"https://image.tmdb.org/t/p/w200{actor['profile_path']}", width=200)
                    st.caption(f"{actor['name']} as {actor.get('character', 'Unknown')}")

        st.markdown("---")
        st.subheader("🎞️ Official Trailer")
        trailer_key = get_content_trailer(content_id, api_type)
        if trailer_key:
            st.video(f"https://www.youtube.com/watch?v={trailer_key}")
        else:
            st.info("No official trailer available.")


        # Streaming Availability (Watch Providers)
        st.markdown("---")
        st.subheader("📺 Streaming Availability")
        try:
            r = requests.get(
                f"{TMDB_BASE}/{api_type}/{content_id}/watch/providers",
                params={"api_key": TMDB_API_KEY},
                timeout=10
            )
            r.raise_for_status()
            providers = r.json().get("results", {}).get("IN") or r.json().get("results", {}).get("US") or {}
            flatrate = providers.get("flatrate", [])
            if flatrate:
                logos = [f"https://image.tmdb.org/t/p/w200{p['logo_path']}" for p in flatrate if p.get("logo_path")]
                cols = st.columns(len(logos))
                for col, logo in zip(cols, logos):
                    col.image(logo, width=60)
            else:
                st.info("Not available on major streaming platforms (TMDb data).")
        except Exception:
            st.warning("Could not fetch streaming availability.")



        # -----------------------
        # Actor filmography explorer (clickable)
        # -----------------------
        st.markdown("---")
        st.subheader("🎬 Actor Filmography Explorer")
        if cast_list:
            selected_actor = st.selectbox(
                "Pick an actor to explore their top movies/TV:",
                [a["name"] for a in cast_list]
            )
            chosen_actor = next((a for a in cast_list if a["name"] == selected_actor), None)
            if chosen_actor:
                try:
                    r = requests.get(
                        f"{TMDB_BASE}/person/{chosen_actor['id']}/movie_credits",
                        params={"api_key": TMDB_API_KEY, "language": "en-US"},
                        timeout=10
                    )
                    r.raise_for_status()
                    data = r.json()
                    top_movies = sorted(
                        data.get("cast", []) or [],
                        key=lambda x: x.get("popularity", 0),
                        reverse=True
                    )[:6]
                except Exception:
                    top_movies = []

                if top_movies:
                    film_cols = st.columns(len(top_movies))
                    for col, m in zip(film_cols, top_movies):
                        with col:
                            if m.get("poster_path"):
                                if st.button(f"🎬 {m.get('title')}", key=f"actor_{m['id']}"):
                                    st.session_state.selected_movie_id = m["id"]
                                    st.session_state.selected_content_type = "movie"
                                st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}", width=180)
                            st.caption(f"{m.get('title')} ({m.get('release_date','')[:4]}) ⭐ {m.get('vote_average',0)}/10")
                else:
                    st.info("No filmography available for this actor.")


# -------------------------------
# Insights Tab
# -------------------------------
with tab2:
    if not details:
        st.info("Search or select an item to see audience insights here.")
    else:
        if not reviews:
            st.info("No reviews found on TMDb.")
        else:
            pos, neg = analyze_sentiment_batch(reviews)
            st.subheader("Sentiment Analysis")
            pie_fig = go.Figure(data=[go.Pie(labels=["Positive", "Negative"], values=[pos, neg], hole=0.45)])
            st.plotly_chart(pie_fig, use_container_width=True)

            st.subheader("📖 Review Summary")
            try:
                summary = summarize_reviews(reviews, max_length=100)
                st.success(summary)
            except Exception:
                st.info("Summary model not available or failed.")

            st.markdown("---")
            st.subheader("☁️ WordCloud (audience)")
            cleaned_tokens = simple_tokenize(" ".join(reviews[:6] or []))
            if cleaned_tokens:
                filtered_text = " ".join(cleaned_tokens)
                wc = WordCloud(width=900, height=450, background_color="black", colormap="plasma", max_words=40).generate(filtered_text)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("Not enough text to build a word cloud.")

# -------------------------------
# Analytics Tab
# -------------------------------
with tab3:
    st.subheader("📊 Analytics")

    if details:  # ✅ only run if details exist
        st.subheader("Director & Actor Influence")
        if director_obj:
            avg_rev, avg_rat, count = analyze_person(director_obj["id"], "Director")
            if count > 0:
                st.write(f"🎬 {director_obj['name']} → Avg Revenue: ${avg_rev}M, Avg Rating: {avg_rat}/10")
        if cast_list:
            lead_actor = cast_list[0]
            avg_rev, avg_rat, count = analyze_person(lead_actor["id"], "Actor")
            if count > 0:
                st.write(f"🎭 {lead_actor['name']} → Avg Revenue: ${avg_rev}M, Avg Rating: {avg_rat}/10")

        st.markdown("---")
        st.subheader("📈 Release Year Trends")
        if director_obj:
            trend_data = get_director_trend(director_obj["id"])
            if trend_data:
                years = [t[0] for t in trend_data]
                revenues = [t[1] / 1e6 for t in trend_data]
                trend_fig = go.Figure()
                trend_fig.add_trace(go.Scatter(x=years, y=revenues, mode="lines+markers"))
                trend_fig.update_layout(
                    title=f"{director_obj['name']}'s Box Office Trend",
                    xaxis_title="Year", yaxis_title="Revenue (Millions $)"
                )
                st.plotly_chart(trend_fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Genre Revenue Trends")
        if details.get("genres"):
            genre_revs = {}
            for g in details["genres"]:
                revenues = get_genre_movies(g.get("id"), pages=2)
                if revenues:
                    avg_rev = np.mean(revenues) / 1e6
                    genre_revs[g["name"]] = round(avg_rev, 2)
            if genre_revs:
                fig = go.Figure([go.Bar(x=list(genre_revs.keys()), y=list(genre_revs.values()), marker_color="orange")])
                fig.update_layout(title="Average Revenue by Genre (Millions $)", yaxis_title="Revenue (M)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No revenue data found for this content's genres.")
        else:
            st.info("No genre data available for this content.")

        st.markdown("---")
        st.subheader("🌍 Regional Popularity Heatmap")
        base_popularity = details.get("popularity", 50)
        countries = ["USA", "IND", "GBR", "FRA", "DEU", "BRA", "JPN", "KOR", "CAN", "AUS"]
        popularity_scores = {c: max(0, float(np.random.normal(loc=base_popularity, scale=10))) for c in countries}
        choropleth_fig = go.Figure(data=go.Choropleth(
            locations=list(popularity_scores.keys()),
            z=list(popularity_scores.values()),
            locationmode="ISO-3",
            colorscale="Blues",
            colorbar_title="Popularity Index"
        ))
        choropleth_fig.update_layout(
            title=f"Simulated Regional Popularity for {details.get('title') or details.get('name')}",
            geo=dict(showframe=False, showcoastlines=True)
        )
        st.plotly_chart(choropleth_fig, use_container_width=True)

    else:
        st.info("ℹ️ Search for a movie/series to see analytics.")

# -------------------------------
# Predictions Tab
# -------------------------------
with tab4:
    if not details:
        st.info("Search or select an item to see predictions here.")
    else:
        st.subheader("💰 Box Office Prediction (demo)")
        budget = details.get("budget", 50) / 1e6 if details.get("budget") else 50 / 1e6
        popularity = details.get("popularity", 50)
        runtime = details.get("runtime", details.get("episode_run_time", [100])[0] if details.get("episode_run_time") else 100)
        vote_avg = details.get("vote_average", 7.0)
        predicted_rev = predict_box_office(budget, popularity, runtime, vote_avg)
        actual_rev = details.get("revenue", 0) / 1e6 if details.get("revenue") else 0
        if actual_rev > 0:
            st.success(f"Predicted: ${predicted_rev}M | Actual: ${round(actual_rev,2)}M")
        else:
            st.info(f"Predicted Worldwide Box Office (estimate): ${predicted_rev}M")

# -------------------------------
# Recommendations Tab (clickable, respects blocklist & adult toggle)
# -------------------------------
with tab5:
    if not details:
        st.info("Search or select an item to see recommendations here.")
    else:
        st.subheader("🎯 Custom Similar Recommendations")
        popular_items = fetch_popular_safe(tmdb_type=api_type, pages=2, include_adult=st.session_state.include_adult)
        scored = []
        for m in popular_items:
            if m.get("id") == details.get("id"):
                continue
            if st.session_state.hide_blocked_recs and is_blocked_by_keywords(m, st.session_state.reco_blocklist):
                continue
            sim = calculate_similarity(details, m)
            scored.append((sim, m))
        scored.sort(reverse=True, key=lambda x: x[0])
        top_recs = [m for _, m in scored[:6]]
        if top_recs:
            rec_cols = st.columns(len(top_recs))
            for col, rec in zip(rec_cols, top_recs):
                with col:
                    title = rec.get("title") or rec.get("name") or "Unknown"
                    if rec.get("poster_path"):
                        # clicking sets selected id/type and Streamlit will rerun
                        if st.button(f"🎬 {title}", key=f"rec_{rec['id']}"):
                            st.session_state.selected_movie_id = rec["id"]
                            st.session_state.selected_content_type = "movie" if api_type == "movie" else "tv"
                        st.image(f"https://image.tmdb.org/t/p/w200{rec['poster_path']}", width=180)
                    st.caption(f"{title} ({(rec.get('release_date') or rec.get('first_air_date') or '')[:4]}) ⭐ {rec.get('vote_average',0)}/10")
        else:
            st.info("No recommendations available based on current filters.")

# -----------------------
# Now Playing Tab (clickable + trending)
# -----------------------
with tab6:
    st.subheader("📅 Movies Now Playing in Theatres")
    r = requests.get(
        f"{TMDB_BASE}/movie/now_playing",
        params={"api_key": TMDB_API_KEY, "language": "en-US", "page": 1},
        timeout=10
    )
    r.raise_for_status()
    now_playing = r.json().get("results", []) or []
    now_playing = _filter_adult(now_playing, st.session_state.include_adult)
    now_playing = now_playing[:10]

    if now_playing:
        cols = st.columns(5)
        for i, m in enumerate(now_playing):
            col = cols[i % 5]
            with col:
                if m.get("poster_path"):
                    if st.button(f"🎬 {m['title']}", key=f"now_{m['id']}"):
                        st.session_state.selected_movie_id = m["id"]
                        st.session_state.selected_content_type = "movie"
                    st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}", width=160)

                st.caption(f"{m['title']} ({m.get('release_date','')[:4]}) ⭐ {m.get('vote_average',0)}/10")

                # 🎟 Ticket booking redirect
                ticket_url = make_bookmyshow_url(m['title'])
                st.markdown(f"[🎟 Book Tickets on BookMyShow]({ticket_url})", unsafe_allow_html=True)
    else:
        st.info("No 'Now Playing' data available.")

    st.markdown("---")
    st.subheader("🔥 Currently Trending (Global)")
    trending_items = fetch_popular_safe("movie", pages=1, include_adult=st.session_state.include_adult)[:8]
    if trending_items:
        cols = st.columns(4)
        for i, m in enumerate(trending_items):
            col = cols[i % 4]
            with col:
                if m.get("poster_path"):
                    if st.button(f"{m.get('title')}", key=f"trend_{m['id']}"):
                        st.session_state.selected_movie_id = m["id"]
                        st.session_state.selected_content_type = "movie"
                    st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}", width=160)
                st.caption(f"{m.get('title')} ({m.get('release_date','')[:4]}) ⭐ {m.get('vote_average',0)}/10")
    else:
        st.info("No trending data available.")

# -------------------------------
# Indian Movies Tab (clickable, filtered to Indian origin via discover)
# -------------------------------
with tab7:
    st.subheader("🎥 Indian Movies Dashboard")

    # Now playing in India
    st.markdown("### 📅 Now Playing in India")
    india_now_playing = get_indian_movies("now_playing", pages=2) or []
    india_now_playing = _filter_adult(india_now_playing, st.session_state.include_adult)[:10]

    if india_now_playing:
        cols = st.columns(5)
        for i, m in enumerate(india_now_playing):
            col = cols[i % 5]
            with col:
                title = m.get("title") or "Unknown"
                if m.get("poster_path"):
                    if st.button(f"🎬 {title}", key=f"in_{m['id']}"):
                        st.session_state.selected_movie_id = m["id"]
                        st.session_state.selected_content_type = "movie"
                    st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}", width=160)
                st.caption(f"{title} ({m.get('release_date','')[:4]}) ⭐ {m.get('vote_average',0)}/10")
                ticket_url = make_bookmyshow_url(title)
                st.markdown(f"[🎟 Book Tickets on BookMyShow]({ticket_url})", unsafe_allow_html=True)
    else:
        st.info("No 'Now Playing' data available for India.")

    # Upcoming in India
    st.markdown("---")
    st.markdown("### 🎬 Upcoming Movies in India")
    india_upcoming = get_indian_movies("upcoming", pages=2) or []
    india_upcoming = _filter_adult(india_upcoming, st.session_state.include_adult)[:10]

    if india_upcoming:
        cols = st.columns(5)
        for i, m in enumerate(india_upcoming):
            col = cols[i % 5]
            with col:
                title = m.get("title") or "Unknown"
                if m.get("poster_path"):
                    if st.button(f"🎬 {title}", key=f"in_up_{m['id']}"):
                        st.session_state.selected_movie_id = m["id"]
                        st.session_state.selected_content_type = "movie"
                    st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}", width=160)
                st.caption(f"{title} ({m.get('release_date','')[:4]})")
    else:
        st.info("No 'Upcoming' data available for India.")

    # Top rated in India
    st.markdown("---")
    st.markdown("### 🏆 Top Rated Indian Movies")
    india_top_rated = get_indian_movies("top_rated", pages=2) or []
    india_top_rated = _filter_adult(india_top_rated, st.session_state.include_adult)[:10]

    if india_top_rated:
        cols = st.columns(5)
        for i, m in enumerate(india_top_rated):
            col = cols[i % 5]
            with col:
                title = m.get("title") or "Unknown"
                if m.get("poster_path"):
                    if st.button(f"🎬 {title}", key=f"in_top_{m['id']}"):
                        st.session_state.selected_movie_id = m["id"]
                        st.session_state.selected_content_type = "movie"
                    st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}", width=160)
                st.caption(f"{title} ⭐ {m.get('vote_average',0)}/10")
    else:
        st.info("No 'Top Rated' data available for India.")


