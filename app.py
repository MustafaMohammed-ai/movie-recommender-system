import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch — Hybrid Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #0d0f14;
    --surface: #161a22;
    --border:  #242836;
    --accent:  #e8b84b;
    --accent2: #5b8dee;
    --text:    #e8e6df;
    --muted:   #7a7d8a;
    --radius:  12px;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.8rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif !important;
    font-size: clamp(2.4rem, 5vw, 3.8rem) !important;
    font-weight: 900 !important;
    letter-spacing: -1px;
    color: var(--accent) !important;
    margin: 0 0 .4rem !important;
}
.hero p { color: var(--muted) !important; font-size: 1rem; margin: 0; }

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text);
    border-left: 4px solid var(--accent);
    padding-left: .75rem;
    margin: 1.5rem 0 1rem;
}

.stat-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: .8rem; }
.stat-pill {
    background: var(--border);
    border-radius: 999px;
    padding: .3rem .8rem;
    font-size: .78rem;
    color: var(--muted);
    white-space: nowrap;
}
.stat-pill b { color: var(--accent); }

.weight-badge {
    background: var(--border);
    border-radius: var(--radius);
    padding: .7rem 1rem;
    margin-top: .5rem;
    font-size: .82rem;
    color: var(--muted);
    line-height: 1.9;
}

.movie-grid { display: flex; flex-direction: column; gap: .65rem; }
.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .85rem 1.1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: border-color .18s;
}
.movie-card:hover { border-color: var(--accent); }
.movie-rank {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 900;
    color: var(--muted);
    min-width: 2rem;
    text-align: center;
    line-height: 1;
    opacity: .5;
}
.movie-rank.top3 { color: var(--accent); opacity: 1; }
.movie-info { flex: 1; min-width: 0; }
.movie-title {
    font-weight: 500;
    font-size: .97rem;
    color: var(--text);
    margin-bottom: .2rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.movie-genres { font-size: .73rem; color: var(--muted); }
.pill {
    display: inline-block;
    background: var(--border);
    color: var(--muted);
    border-radius: 999px;
    padding: .12rem .5rem;
    font-size: .68rem;
    margin-right: .2rem;
    margin-bottom: .15rem;
}
.movie-right { text-align: right; min-width: 72px; }
.movie-score {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.score-bar-bg {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 5px;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--accent);
}

.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--muted);
}
.empty-state .icon { font-size: 3rem; margin-bottom: .8rem; }
.empty-state h3 {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    color: var(--text);
    margin: 0 0 .5rem;
}
.empty-state p { font-size: .9rem; margin: 0; }

.rated-table { width: 100%; border-collapse: collapse; }
.rated-table th {
    background: var(--border);
    padding: .45rem .75rem;
    text-align: left;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--muted);
}
.rated-table td {
    padding: .48rem .75rem;
    border-bottom: 1px solid var(--border);
    font-size: .88rem;
    color: var(--text);
}
.stars { color: var(--accent); letter-spacing: .05em; font-size: .85rem; }
.rating-num { color: var(--muted); font-size: .78rem; margin-left: .2rem; }

.metric-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-top: .9rem; }
.metric-card {
    flex: 1 1 100px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem 1.1rem;
    text-align: center;
}
.metric-card .val {
    font-family: 'Playfair Display', serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-card .lbl {
    font-size: .68rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: .08em;
    margin-top: .3rem;
}

.perf-table { width: 100%; border-collapse: collapse; }
.perf-table th {
    padding: .55rem .8rem;
    background: var(--border);
    text-align: left;
    font-size: .72rem;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--muted);
}
.perf-table td { padding: .48rem .8rem; border-bottom: 1px solid var(--border); font-size: .9rem; }
.perf-table tr:last-child td { border-bottom: none; }
.model-name { color: var(--text); font-weight: 500; }
.val-best  { color: #5ecb8a; font-weight: 600; }
.val-ok    { color: var(--accent); }

.stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 700 !important;
    padding: .65rem 2rem !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    width: 100%;
}
.stButton > button:hover { opacity: .88 !important; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

RATING_SCALE = (0.5, 5.0)


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _clip(v):
    return float(np.clip(v, *RATING_SCALE))


def _stars(rating: float) -> str:
    full = int(rating)
    half = (rating - full) >= 0.5
    return ("★" * full) + ("½" if half else "")


def _bar_pct(score: float) -> int:
    return int(round((score - 0.5) / 4.5 * 100))


# ══════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    movies  = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    tags    = pd.read_csv("tags.csv")

    for df in [movies, ratings, tags]:
        df.drop_duplicates(inplace=True)

    movies.fillna("", inplace=True)
    ratings.dropna(inplace=True)
    tags.fillna("", inplace=True)

    ratings["userId"]  = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    ratings["rating"]  = ratings["rating"].astype(float)
    movies["movieId"]  = movies["movieId"].astype(int)

    # merge tags
    movie_tags = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x))
        .reset_index()
        .rename(columns={"tag": "tags_text"})
    )

    movies = movies.merge(movie_tags, on="movieId", how="left")
    movies["tags_text"] = movies["tags_text"].fillna("")

    movies["content"] = (
        movies["title"] + " " +
        movies["genres"].str.replace("|", " ", regex=False) + " " +
        movies["tags_text"]
    )

    return movies, ratings


# ══════════════════════════════════════════════
# BUILD MODELS
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_models(_movies, _ratings):

    # CONTENT MODEL
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(_movies["content"])
    cosine_sim_content = cosine_similarity(tfidf_matrix, tfidf_matrix)

    id2idx = pd.Series(_movies.index, index=_movies["movieId"]).drop_duplicates()

    # COLLABORATIVE MODEL (Item-Item similarity)
    rating_matrix = _ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    cosine_sim_items = cosine_similarity(rating_matrix.T)

    return cosine_sim_content, id2idx, rating_matrix, cosine_sim_items


# ══════════════════════════════════════════════
# CONTENT PREDICT
# ══════════════════════════════════════════════
def content_predict(uid, mid, ratings, cosine_sim, id2idx):
    user_movies = ratings[ratings["userId"] == uid]

    if user_movies.empty:
        return 0.0

    if mid not in id2idx:
        return float(user_movies["rating"].mean())

    target_idx = id2idx[mid]

    weighted_scores = []
    sim_values = []

    for _, row in user_movies.iterrows():
        rated_mid = row["movieId"]

        if rated_mid not in id2idx:
            continue

        rated_idx = id2idx[rated_mid]
        sim = cosine_sim[target_idx][rated_idx]

        weighted_scores.append(sim * row["rating"])
        sim_values.append(sim)

    total_sim = sum(sim_values)

    if total_sim == 0:
        return float(user_movies["rating"].mean())

    return sum(weighted_scores) / total_sim


# ══════════════════════════════════════════════
# COLLABORATIVE PREDICT (Item-Item)
# ══════════════════════════════════════════════
def collaborative_predict(uid, mid, rating_matrix, cosine_sim_items):
    if uid not in rating_matrix.index:
        return 0.0
    if mid not in rating_matrix.columns:
        return float(rating_matrix.loc[uid].replace(0, np.nan).mean())

    movie_list = list(rating_matrix.columns)
    mid_index = movie_list.index(mid)

    sim_scores = cosine_sim_items[mid_index]
    user_ratings = rating_matrix.loc[uid].values

    weighted_sum = np.dot(sim_scores, user_ratings)
    sim_sum = np.sum(np.abs(sim_scores))

    if sim_sum == 0:
        return float(rating_matrix.loc[uid].replace(0, np.nan).mean())

    return weighted_sum / sim_sum


# ══════════════════════════════════════════════
# HYBRID PREDICT
# ══════════════════════════════════════════════
def hybrid_predict(uid, mid, ratings, cosine_sim, id2idx,
                   rating_matrix, cosine_sim_items, alpha=0.4):

    content_score = _clip(content_predict(uid, mid, ratings, cosine_sim, id2idx))
    collab_score  = _clip(collaborative_predict(uid, mid, rating_matrix, cosine_sim_items))

    return alpha * content_score + (1 - alpha) * collab_score


# ══════════════════════════════════════════════
# RECOMMEND MOVIES
# ══════════════════════════════════════════════
def recommend_movies(uid, movies, ratings, cosine_sim, id2idx,
                     rating_matrix, cosine_sim_items, top_n=10, alpha=0.4):

    rated = set(ratings[ratings["userId"] == uid]["movieId"].values)
    all_movies = movies["movieId"].unique()

    preds = []
    for mid in all_movies:
        if mid in rated:
            continue

        score = hybrid_predict(uid, mid, ratings, cosine_sim, id2idx,
                               rating_matrix, cosine_sim_items, alpha)
        preds.append((mid, score))

    preds.sort(key=lambda x: x[1], reverse=True)

    result = pd.DataFrame(preds[:top_n], columns=["movieId", "hybrid_score"])
    result = result.merge(movies[["movieId", "title", "genres"]], on="movieId")
    result["hybrid_score"] = result["hybrid_score"].round(3)

    return result[["title", "genres", "hybrid_score"]]


# ══════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_metrics(ratings, cosine_sim, id2idx, rating_matrix, cosine_sim_items):

    test_df = ratings.sample(frac=0.2, random_state=42)

    y_true = []
    y_content = []
    y_collab = []
    y_hybrid = []

    for _, row in test_df.iterrows():
        uid = int(row["userId"])
        mid = int(row["movieId"])
        actual = float(row["rating"])

        c = _clip(content_predict(uid, mid, ratings, cosine_sim, id2idx))
        v = _clip(collaborative_predict(uid, mid, rating_matrix, cosine_sim_items))
        h = _clip(0.4 * c + 0.6 * v)

        y_true.append(actual)
        y_content.append(c)
        y_collab.append(v)
        y_hybrid.append(h)

    def _calc(y_t, y_p):
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mae = mean_absolute_error(y_t, y_p)

        threshold = 3.5
        tp = fp = fn = 0

        for a, p in zip(y_t, y_p):
            if p >= threshold and a >= threshold:
                tp += 1
            elif p >= threshold and a < threshold:
                fp += 1
            elif p < threshold and a >= threshold:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        return rmse, mae, precision, recall, f1

    return _calc(y_true, y_content), _calc(y_true, y_collab), _calc(y_true, y_hybrid)


# ══════════════════════════════════════════════
# LOAD EVERYTHING
# ══════════════════════════════════════════════
with st.spinner("🎬 Loading models..."):
    movies, ratings = load_data()
    cosine_sim, id2idx, rating_matrix, cosine_sim_items = build_models(movies, ratings)


# ══════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>Hybrid Movie Recommendation · Content-Based + Collaborative Filtering</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    selected_user = st.selectbox("👤 User ID", sorted(ratings["userId"].unique()), index=0)
    top_n = st.slider("🎯 Recommendations", 5, 20, 10)
    alpha = st.slider("⚖️ Content Weight (α)", 0.0, 1.0, 0.4, 0.05)

    st.markdown(f"""
    <div class="weight-badge">
        <b style="color:#e8b84b">α = {alpha:.2f}</b> → content-based<br>
        <b style="color:#5b8dee">β = {1-alpha:.2f}</b> → collaborative
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    run_btn = st.button("🚀 Get Recommendations")

    st.markdown("---")
    st.markdown("**📌 Dataset**")
    st.markdown(f"""
    <div class="stat-row">
        <span class="stat-pill">👥 <b>{ratings['userId'].nunique():,}</b> users</span>
        <span class="stat-pill">🎬 <b>{ratings['movieId'].nunique():,}</b> movies</span>
        <span class="stat-pill">⭐ <b>{len(ratings):,}</b> ratings</span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")


# ── Recommendations ──────────────────────────
with col_left:
    if run_btn:
        with st.spinner("Computing recommendations…"):
            recs = recommend_movies(
                selected_user, movies, ratings,
                cosine_sim, id2idx,
                rating_matrix, cosine_sim_items,
                top_n=top_n, alpha=alpha
            )

        st.markdown(
            f'<div class="section-title">Top {top_n} picks for User #{selected_user}</div>',
            unsafe_allow_html=True,
        )

        cards = ['<div class="movie-grid">']
        for rank, (_, row) in enumerate(recs.iterrows(), start=1):
            title = row["title"]
            score = row["hybrid_score"]
            top_cls = "top3" if rank <= 3 else ""

            pills = "".join(
                f'<span class="pill">{g.strip()}</span>'
                for g in row["genres"].replace("|", ",").split(",") if g.strip()
            )

            cards.append(f"""
            <div class="movie-card">
              <div class="movie-rank {top_cls}">{rank:02d}</div>
              <div class="movie-info">
                <div class="movie-title" title="{title}">{title[:54]}{"…" if len(title)>54 else ""}</div>
                <div class="movie-genres">{pills}</div>
              </div>
              <div class="movie-right">
                <div class="movie-score">{score:.2f}</div>
                <div class="score-bar-bg">
                  <div class="score-bar-fill" style="width:{_bar_pct(score)}%"></div>
                </div>
              </div>
            </div>""")

        cards.append("</div>")
        st.markdown("".join(cards), unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
          <div class="icon">🎬</div>
          <h3>Your personalised picks await</h3>
          <p>Choose a user ID and hit <em>Get Recommendations</em></p>
        </div>
        """, unsafe_allow_html=True)


# ── User History ─────────────────────────────
with col_right:
    st.markdown(
        f'<div class="section-title">📋 User #{selected_user} — History</div>',
        unsafe_allow_html=True,
    )

    user_hist = (
        ratings[ratings["userId"] == selected_user]
        .merge(movies[["movieId", "title"]], on="movieId")
        .sort_values("rating", ascending=False)
        .head(15)
    )

    if user_hist.empty:
        st.info("No ratings found for this user.")
    else:
        rows = []
        for _, r in user_hist.iterrows():
            t = r["title"]
            rows.append(f"""
            <tr>
              <td>{t[:36]}{"…" if len(t)>36 else ""}</td>
              <td><span class="stars">{_stars(r["rating"])}</span>
                  <span class="rating-num">{r["rating"]}</span></td>
            </tr>""")

        st.markdown(f"""
        <table class="rated-table">
          <thead><tr><th>Movie</th><th>Rating</th></tr></thead>
          <tbody>{"".join(rows)}</tbody>
        </table>
        """, unsafe_allow_html=True)

        u_ratings = ratings[ratings["userId"] == selected_user]["rating"]
        high_r = (u_ratings >= 4.0).sum()

        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-card">
            <div class="val">{len(u_ratings)}</div>
            <div class="lbl">Rated</div>
          </div>
          <div class="metric-card">
            <div class="val">{u_ratings.mean():.1f}</div>
            <div class="lbl">Avg ★</div>
          </div>
          <div class="metric-card">
            <div class="val">{high_r}</div>
            <div class="lbl">Loved ≥4★</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PERFORMANCE METRICS
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)

with st.expander("Show evaluation metrics — 20 % test split", expanded=False):
    with st.spinner("Evaluating all models…"):
        content_m, collab_m, hybrid_m = compute_metrics(
            ratings, cosine_sim, id2idx, rating_matrix, cosine_sim_items
        )

    labels = ["RMSE ↓", "MAE ↓", "Precision ↑", "Recall ↑", "F1 ↑"]
    lower_better = [True, True, False, False, False]

    model_data = [
        ("Content-Based", content_m),
        ("Collaborative", collab_m),
        ("Hybrid", hybrid_m),
    ]

    best = [
        min(m[1][j] for m in model_data) if lb else max(m[1][j] for m in model_data)
        for j, lb in enumerate(lower_better)
    ]

    header = "".join(
        f'<th style="padding:.55rem .8rem;background:#242836;color:#7a7d8a;'
        f'text-align:left;font-size:.72rem;letter-spacing:.07em">{l}</th>'
        for l in ["MODEL"] + labels
    )

    body = ""
    for mname, mvals in model_data:
        cells = f'<td class="model-name" style="padding:.48rem .8rem">{mname}</td>'
        for j, v in enumerate(mvals):
            css = "val-best" if abs(v - best[j]) < 1e-9 else "val-ok"
            cells += f'<td class="{css}" style="padding:.48rem .8rem">{v:.4f}</td>'
        body += f"<tr>{cells}</tr>"

    st.markdown(f"""
    <table class="perf-table"
           style="background:var(--surface);border:1px solid var(--border);
                  border-radius:var(--radius);overflow:hidden">
      <thead><tr>{header}</tr></thead>
      <tbody>{body}</tbody>
    </table>
    <p style="font-size:.75rem;color:#7a7d8a;margin-top:.6rem">
        🟢 Green = best for that metric &nbsp;·&nbsp; Threshold = 3.5 ★
    </p>
    """, unsafe_allow_html=True)
