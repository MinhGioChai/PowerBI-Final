# ========================= app.py =========================

import os
import pickle
import logging
import numpy as np
import pandas as pd

from flask import (
    Flask,
    request,
    redirect,
    url_for
)

from sklearn.metrics.pairwise import linear_kernel


# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hybrid_recommender.pkl"
)

TOP_K = 12


# =========================================================
# LOAD HYBRID MODEL
# =========================================================
logger.info("Loading hybrid recommender...")

with open(MODEL_PATH, "rb") as f:

    bundle = pickle.load(f)


# =========================================================
# LOAD OBJECTS
# =========================================================

# NLP
tfidf = bundle["tfidf"]
tfidf_matrix = bundle["tfidf_matrix"]

# Index
article_to_idx = bundle["article_to_idx"]
idx_to_article = bundle["idx_to_article"]

# History
customer_history = bundle["customer_history"]

# Popular
top_pop = bundle["top_pop"]

# Metadata
articles = bundle["article_meta"].copy()


# =========================================================
# CLEAN DATA
# =========================================================
logger.info("Preparing metadata...")

articles = articles.fillna("")

# purchase_count already exists from training
if "purchase_count" not in articles.columns:

    articles["purchase_count"] = 0

articles["purchase_count"] = pd.to_numeric(
    articles["purchase_count"],
    errors="coerce"
).fillna(0).astype(int)

# pop_score already exists from training
if "pop_score" not in articles.columns:

    max_pop = articles["purchase_count"].max()

    if max_pop > 0:

        articles["pop_score"] = (
            articles["purchase_count"] / max_pop
        )

    else:

        articles["pop_score"] = 0


# =========================================================
# SAFE DISPLAY COLUMNS
# =========================================================
display_cols = [
    "article_id",
    "prod_name",
    "colour_group_name",
    "graphical_appearance_name",
    "garment_group_name",
    "purchase_count"
]

for c in display_cols:

    if c not in articles.columns:

        articles[c] = ""


# =========================================================
# PRECOMPUTE ARRAYS
# =========================================================
logger.info("Precomputing arrays...")

article_ids = articles["article_id"].values

pop_array = (
    articles["pop_score"]
    .astype(float)
    .values
)

color_array = (
    articles["colour_group_name"]
    .astype(str)
    .values
)


# =========================================================
# CUSTOMER COLOR PREFERENCE
# =========================================================
logger.info("Building customer preferences...")

customer_color_pref = {}

for customer_id, history in customer_history.items():

    tmp = articles[
        articles["article_id"].isin(history)
    ]

    if len(tmp) == 0:
        continue

    pref = (
        tmp["colour_group_name"]
        .value_counts(normalize=True)
        .to_dict()
    )

    customer_color_pref[customer_id] = pref


# =========================================================
# HYBRID SCORE
# =========================================================
def compute_score(
    article_id,
    customer_id=None
):

    if article_id not in article_to_idx:
        return None

    idx = article_to_idx[article_id]

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================
    sim = linear_kernel(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    # =====================================================
    # HYBRID SCORE
    # =====================================================
    score = (
        0.45 * sim +
        0.45 * pop_array
    )

    # =====================================================
    # COLOR BEHAVIOR
    # =====================================================
    if customer_id in customer_color_pref:

        pref = customer_color_pref[customer_id]

        color_score = np.array([
            pref.get(c, 0)
            for c in color_array
        ])

        score += 0.10 * color_score

    return score


# =========================================================
# RECOMMENDATION
# =========================================================
def recommend_from_article(
    article_id,
    top_k=10,
    customer_id=None
):

    score = compute_score(
        article_id,
        customer_id
    )

    if score is None:

        return pd.DataFrame()

    # =====================================================
    # GET TOP INDEX
    # =====================================================
    top_idx = np.argpartition(
        -score,
        top_k + 20
    )[:top_k + 20]

    top_idx = top_idx[
        np.argsort(-score[top_idx])
    ]

    rec_ids = []

    for i in top_idx:

        aid = idx_to_article[i]

        if aid == article_id:
            continue

        rec_ids.append(aid)

        if len(rec_ids) >= top_k:
            break

    result = articles[
        articles["article_id"].isin(rec_ids)
    ].copy()

    result["final_score"] = (
        result["article_id"]
        .apply(
            lambda x:
            score[
                article_to_idx[x]
            ]
        )
    )

    result = result.sort_values(
        "final_score",
        ascending=False
    )

    return result


# =========================================================
# CUSTOMER RECOMMENDATION
# =========================================================
def recommend_customer(
    customer_id,
    top_k=10
):

    if customer_id not in customer_history:

        return pd.DataFrame()

    history = customer_history[customer_id]

    if len(history) == 0:

        return pd.DataFrame()

    seed_article = history[-1]

    result = recommend_from_article(
        seed_article,
        top_k=top_k,
        customer_id=customer_id
    )

    result["seed_article"] = seed_article

    return result


# =========================================================
# SEARCH PRODUCTS
# =========================================================
def search_products(
    query,
    top_k=20
):

    q_vec = tfidf.transform([query])

    sim = linear_kernel(
        q_vec,
        tfidf_matrix
    ).flatten()

    final_score = (
        0.75 * sim +
        0.25 * pop_array
    )

    top_idx = np.argsort(
        -final_score
    )[:top_k]

    result = articles.iloc[top_idx].copy()

    result["search_score"] = (
        final_score[top_idx]
    )

    return result


# =========================================================
# LOGIN PAGE
# =========================================================
@app.route("/")
def login():

    return """
    <html>

    <head>
        <title>Fashion AI</title>
    </head>

    <body style="
        font-family:Arial;
        background:#f5f5f5;
        text-align:center;
        margin-top:120px;
    ">

        <div style="
            background:white;
            padding:40px;
            display:inline-block;
            border-radius:18px;
            box-shadow:0 10px 40px rgba(0,0,0,0.08)
        ">

            <h1 style="color:#E50010">
                FASHION AI
            </h1>

            <p>
                Hybrid Recommendation System
            </p>

            <form action="/shop">

                <input
                    name="customer_id"
                    placeholder="Enter Customer ID"
                    style="
                        padding:12px;
                        width:320px;
                        border-radius:10px;
                        border:1px solid #ddd;
                    "
                >

                <br><br>

                <button style="
                    padding:12px 20px;
                    border:none;
                    border-radius:10px;
                    background:#E50010;
                    color:white;
                    font-weight:700;
                    cursor:pointer;
                ">
                    Enter Shop
                </button>

            </form>

        </div>

    </body>

    </html>
    """


# =========================================================
# SHOP
# =========================================================
@app.route("/shop")
def shop():

    customer_id = request.args.get(
        "customer_id",
        ""
    ).strip()

    query = request.args.get(
        "q",
        ""
    ).strip()

    if not customer_id:

        return redirect(
            url_for("login")
        )

    # =====================================================
    # SEARCH
    # =====================================================
    if query:

        recs = search_products(
            query,
            top_k=20
        )

        title = f"Search Results: {query}"

    # =====================================================
    # CUSTOMER RECOMMENDATION
    # =====================================================
    else:

        recs = recommend_customer(
            customer_id,
            top_k=20
        )

        # fallback
        if len(recs) == 0:

            recs = articles.sort_values(
                "purchase_count",
                ascending=False
            ).head(20)

        title = "Recommended For You"

    return render_html(
        customer_id,
        recs,
        title,
        query
    )


# =========================================================
# SIMILAR
# =========================================================
@app.route("/similar/<article_id>")
def similar(article_id):

    customer_id = request.args.get(
        "customer_id",
        ""
    )

    try:
        article_id = int(article_id)

    except:
        return "Invalid article id"

    recs = recommend_from_article(
        article_id,
        top_k=12,
        customer_id=customer_id
    )

    return render_html(
        customer_id,
        recs,
        "Similar Products",
        ""
    )


# =========================================================
# HTML
# =========================================================
def render_html(
    customer_id,
    data,
    title,
    query
):

    cards = ""

    for _, row in data.iterrows():

        cards += f"""

        <div class="card">

            <div class="img">
                H&M
            </div>

            <div class="content">

                <div class="name">
                    {row['prod_name']}
                </div>

                <div class="meta">
                    {row['colour_group_name']}
                    •
                    {row['garment_group_name']}
                </div>

                <div class="meta2">
                    {row['graphical_appearance_name']}
                </div>

                <div class="bottom">

                    <div class="trend">
                        🔥 {row['purchase_count']} purchases
                    </div>

                    <a
                        class="btn"
                        href="/similar/{row['article_id']}?customer_id={customer_id}"
                    >
                        Similar
                    </a>

                </div>

            </div>

        </div>
        """

    return f"""

    <html>

    <head>

        <title>Fashion AI</title>

        <style>

            body {{
                margin:0;
                font-family:Arial;
                background:#f7f7f7;
            }}

            .top {{
                background:white;
                padding:15px 25px;
                display:flex;
                justify-content:space-between;
                align-items:center;
                box-shadow:0 2px 10px rgba(0,0,0,0.05);
            }}

            .logo {{
                color:#E50010;
                font-size:24px;
                font-weight:900;
            }}

            .search input {{
                padding:10px;
                width:300px;
                border-radius:10px;
                border:1px solid #ddd;
            }}

            .grid {{
                display:grid;
                grid-template-columns:
                    repeat(
                        auto-fit,
                        minmax(240px, 1fr)
                    );

                gap:20px;
                padding:20px;
            }}

            .card {{
                background:white;
                border-radius:16px;
                overflow:hidden;
                box-shadow:0 5px 20px rgba(0,0,0,0.06);
                transition:0.2s;
            }}

            .card:hover {{
                transform:translateY(-4px);
            }}

            .img {{
                height:180px;
                background:#ececec;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:28px;
                font-weight:900;
                color:#999;
            }}

            .content {{
                padding:15px;
            }}

            .name {{
                font-size:16px;
                font-weight:700;
                margin-bottom:6px;
            }}

            .meta {{
                color:#666;
                font-size:13px;
            }}

            .meta2 {{
                color:#999;
                font-size:12px;
                margin-top:4px;
            }}

            .bottom {{
                margin-top:14px;
                display:flex;
                justify-content:space-between;
                align-items:center;
            }}

            .trend {{
                font-size:13px;
                color:#E50010;
                font-weight:700;
            }}

            .btn {{
                background:#E50010;
                color:white;
                text-decoration:none;
                padding:8px 12px;
                border-radius:10px;
                font-size:13px;
                font-weight:700;
            }}

            h2 {{
                padding:20px;
                margin:0;
            }}

        </style>

    </head>

    <body>

        <div class="top">

            <div class="logo">
                FASHION AI
            </div>

            <form class="search">

                <input
                    name="q"
                    value="{query}"
                    placeholder="Search products..."
                >

                <input
                    type="hidden"
                    name="customer_id"
                    value="{customer_id}"
                >

            </form>

        </div>

        <h2>{title}</h2>

        <div class="grid">
            {cards}
        </div>

    </body>

    </html>
    """


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":

    logger.info(
        "Running app at http://127.0.0.1:5000"
    )

    app.run(
        debug=True
    )