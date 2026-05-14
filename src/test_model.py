import os
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix
)

from sklearn.metrics.pairwise import linear_kernel


# CONFIG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_PATH = os.path.join(BASE_DIR, "test_data.csv")
TRAIN_PATH = os.path.join(BASE_DIR, "train_data.csv")
ARTICLES_PATH = os.path.join(BASE_DIR, "articles_with_desc.csv")

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "hybrid_recommender.pkl"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "test"
)

TOP_K = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD DATA
logger.info("Loading datasets...")

test = pd.read_csv(TEST_PATH)
train = pd.read_csv(TRAIN_PATH)
articles = pd.read_csv(ARTICLES_PATH)

print("Test shape:", test.shape)
print("Train shape:", train.shape)
print("Articles shape:", articles.shape)

if "target" not in test.columns:
    raise ValueError("test_data.csv must contain target column")


# LOAD HYBRID MODEL
logger.info("Loading hybrid recommender...")

with open(MODEL_PATH, "rb") as f:

    bundle = pickle.load(f)

xgb_model = bundle["xgb_model"]

tfidf = bundle["tfidf"]
svd = bundle["svd"]

tfidf_matrix = bundle["tfidf_matrix"]

article_to_idx = bundle["article_to_idx"]
idx_to_article = bundle["idx_to_article"]

customer_history = bundle["customer_history"]

top_pop = bundle["top_pop"]

feature_cols = bundle["feature_cols"]
svd_cols = bundle["svd_cols"]


# BUILD ARTICLE TEXT
logger.info("Building article text...")

articles["text"] = (
    articles["prod_name"].fillna("") + " " +
    articles["product_type_name"].fillna("") + " " +
    articles["colour_group_name"].fillna("") + " " +
    articles["section_name"].fillna("") + " " +
    articles["garment_group_name"].fillna("") + " " +
    articles["clothes_description"].fillna("")
)


# ARTICLE POPULARITY
logger.info("Building purchase counts...")

article_popularity = (
    train[train["target"] == 1]
    .groupby("article_id")
    .size()
    .reset_index(name="purchase_count")
)

articles = articles.merge(
    article_popularity,
    on="article_id",
    how="left"
)

articles["purchase_count"] = (
    articles["purchase_count"]
    .fillna(0)
    .astype(int)
)


# TFIDF + SVD
logger.info("Transforming article text...")

tfidf_features = tfidf.transform(
    articles["text"]
)

svd_matrix = svd.transform(
    tfidf_features
)

svd_df = pd.DataFrame(
    svd_matrix,
    columns=svd_cols
)

articles = pd.concat(
    [articles.reset_index(drop=True), svd_df],
    axis=1
)


# MERGE ARTICLE FEATURES
logger.info("Merging article features...")

rows_before_merge = len(test)

test = test.merge(
    articles[["article_id"] + svd_cols],
    on="article_id",
    how="left"
)

print("\n==============================")
print("MERGE DIAGNOSTIC")
print("==============================")

print("Rows before merge:",
      rows_before_merge)

print("Rows after merge :",
      len(test))

print("Missing SVD rows :",
      test[svd_cols].isna().all(axis=1).mean())


# HANDLE MISSING
logger.info("Handling missing values...")

zero_fill_cols = [
    "recency_days",
    "purchase_frequency",
    "unique_articles",
    "total_spent",
    "is_expensive_for_user"
]

median_fill_cols = [
    "avg_purchase_value",
    "article_popularity",
    "article_avg_price",
    "log_popularity",
    "log_unique_customers",
    "price_diff"
]

pref_cols = [
    "favorite_index_encoded",
    "favorite_color_encoded"
]

# Fill SVD
test[svd_cols] = test[svd_cols].fillna(0)

# Fill preference cols
for c in pref_cols:

    if c in test.columns:

        test[c] = test[c].fillna(-1)

# Fill zero cols
for c in zero_fill_cols:

    if c in test.columns:

        test[c] = test[c].fillna(0)

# Fill median cols
for c in median_fill_cols:

    if c in test.columns and c in train.columns:

        median_value = train[c].median()

        test[c] = test[c].fillna(
            median_value
        )

# Drop missing target
test = test.dropna(
    subset=["target"]
).copy()

print("\n==============================")
print("MISSING AFTER FILL")
print("==============================")

print("Remaining missing:",
      test[feature_cols].isna().sum().sum())

print("Final test shape:",
      test.shape)


# TEST DATA
X_test = test[feature_cols]
y_test = test["target"]

print("\nTarget rate:",
      y_test.mean())


# XGBOOST PREDICTION
logger.info("Running XGBoost test...")

preds = xgb_model.predict_proba(
    X_test
)[:, 1]

pred_label = (
    preds >= 0.5
).astype(int)


# METRICS
pr_auc = average_precision_score(
    y_test,
    preds
)

roc_auc = roc_auc_score(
    y_test,
    preds
)

ll = log_loss(
    y_test,
    preds
)

precision = precision_score(
    y_test,
    pred_label
)

recall = recall_score(
    y_test,
    pred_label
)

f2 = fbeta_score(
    y_test,
    pred_label,
    beta=2
)

print("\n==============================")
print("XGBOOST TEST")
print("==============================")

print(f"PR AUC    : {pr_auc:.5f}")
print(f"ROC AUC   : {roc_auc:.5f}")
print(f"LogLoss   : {ll:.5f}")
print(f"Precision : {precision:.5f}")
print(f"Recall    : {recall:.5f}")
print(f"F2        : {f2:.5f}")


# CONFUSION MATRIX
tn, fp, fn, tp = confusion_matrix(
    y_test,
    pred_label
).ravel()

confusion = pd.DataFrame([{

    "True Positive": tp,
    "True Negative": tn,
    "False Positive": fp,
    "False Negative": fn
}])


# PREDICTIONS
predictions = test[
    [
        "customer_id",
        "article_id",
        "target"
    ]
].copy()

predictions["predicted_probability"] = preds
predictions["predicted_label"] = pred_label


# FEATURE IMPORTANCE
feature_importance = pd.DataFrame({

    "feature": feature_cols,

    "importance":
        xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "importance",
    ascending=False
)


# TOP-K RECOMMENDATIONS
topk = predictions.copy()

topk["temp_rank"] = (

    topk.groupby("customer_id")[
        "predicted_probability"
    ]

    .rank(
        method="first",
        ascending=False
    )
)

topk = topk[
    topk["temp_rank"] <= TOP_K
].copy()

topk = topk.drop(
    columns=["temp_rank"]
)

topk = topk.merge(
    articles[
        [
            "article_id",
            "prod_name",
            "purchase_count"
        ]
    ],
    on="article_id",
    how="left"
)


# DISPLAY COLUMNS
display_cols = [
    "article_id",
    "prod_name",
    "colour_group_name",
    "graphical_appearance_name",
    "garment_group_name",
    "purchase_count"
]


# COLOR PREFERENCE
def get_color_pref(customer_id):

    if customer_id not in customer_history:

        return None

    history = customer_history[customer_id]

    hist_df = articles[
        articles["article_id"].isin(history)
    ]

    if len(hist_df) == 0:

        return None

    return hist_df[
        "colour_group_name"
    ].mode().iloc[0]


# CONTENT-BASED RECOMMENDATION
def recommend_from_article(
    article_id,
    top_k=10,
    customer_id=None
):

    if article_id not in article_to_idx:

        return pd.DataFrame()

    idx = article_to_idx[article_id]

    sim = linear_kernel(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    color_pref = (
        get_color_pref(customer_id)
        if customer_id else None
    )

    results = []

    for i in np.argsort(sim)[::-1]:

        aid = idx_to_article[i]

        if aid == article_id:

            continue

        row = articles[
            articles["article_id"] == aid
        ]

        if row.empty:

            continue

        row = row.iloc[0]

        sim_score = sim[i]

        purchase_count = row[
            "purchase_count"
        ]

        pop_score = (
            purchase_count /
            (articles["purchase_count"].max() + 1)
        )

        color_score = (

            1.0

            if (
                color_pref and
                row["colour_group_name"] == color_pref
            )

            else 0.0
        )

        final_score = (
            0.65 * sim_score +
            0.25 * pop_score +
            0.10 * color_score
        )

        results.append(
            (aid, final_score)
        )

    results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    rec_ids = [r[0] for r in results]

    result_df = articles[
        articles["article_id"].isin(rec_ids)
    ].copy()

    return result_df

# OLD CUSTOMER TEST
logger.info("Testing old customer recommendation...")

old_customer_id = list(
    customer_history.keys()
)[0]

seed_article = customer_history[
    old_customer_id
][-1]

old_result = recommend_from_article(
    seed_article,
    top_k=TOP_K,
    customer_id=old_customer_id
)

print("\n==============================")
print("OLD CUSTOMER TEST")
print("==============================")

print("Customer:",
      old_customer_id)

print("Seed article:",
      seed_article)

print(old_result[
    display_cols
].to_string(index=False))


# NEW CUSTOMER TEST
logger.info("Testing new customer recommendation...")

seed_article = top_pop[0]

new_result = recommend_from_article(
    seed_article,
    top_k=TOP_K,
    customer_id=None
)

print("\n==============================")
print("NEW CUSTOMER TEST")
print("==============================")

print("Seed article:",
      seed_article)

print(new_result[
    display_cols
].to_string(index=False))


# SAVE SUMMARY
summary = pd.DataFrame([{

    "Model": "XGBoost",

    "Precision": precision,
    "Recall": recall,
    "F2": f2,
    "PR_AUC": pr_auc,
    "ROC_AUC": roc_auc,
    "LogLoss": ll,

    "Target_Rate": y_test.mean(),

    "Rows": len(test)
}])


# SAVE FILES
summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_model_summary.csv"
    ),
    index=False
)

predictions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    ),
    index=False
)

topk.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "topk_recommendations.csv"
    ),
    index=False
)

feature_importance.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "feature_importance.csv"
    ),
    index=False
)

confusion.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    ),
    index=False
)

old_result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "old_customer_recommendation.csv"
    ),
    index=False
)

new_result.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "new_customer_recommendation.csv"
    ),
    index=False
)

logger.info("Hybrid XGBoost test completed")