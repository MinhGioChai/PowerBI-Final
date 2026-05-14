import os
import pickle
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    roc_auc_score,
    log_loss
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


# CONFIG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(BASE_DIR, "train_data.csv")
ARTICLES_PATH = os.path.join(BASE_DIR, "articles_with_desc.csv")

MODEL_DIR = os.path.join(BASE_DIR, "model")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOAD DATA
logger.info("Loading dataset...")

df = pd.read_csv(TRAIN_PATH)
articles = pd.read_csv(ARTICLES_PATH)

print("Train shape:", df.shape)
print("Articles shape:", articles.shape)


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


# TF-IDF
logger.info("Training TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(
    articles["text"]
)


# SVD
logger.info("Running SVD...")

svd = TruncatedSVD(
    n_components=64,
    random_state=42
)

svd_matrix = svd.fit_transform(
    tfidf_matrix
)

svd_cols = [f"svd_{i}" for i in range(64)]

svd_df = pd.DataFrame(
    svd_matrix,
    columns=svd_cols
)

articles = pd.concat(
    [articles.reset_index(drop=True), svd_df],
    axis=1
)


# ARTICLE INDEX
article_ids = articles["article_id"].tolist()

article_to_idx = {
    a: i for i, a in enumerate(article_ids)
}

idx_to_article = {
    i: a for i, a in enumerate(article_ids)
}


# POPULARITY
logger.info("Building popularity features...")

article_pop = (
    df[df["target"] == 1]
    .groupby("article_id")
    .size()
    .reset_index(name="purchase_count")
)

articles = articles.merge(
    article_pop,
    on="article_id",
    how="left"
)

articles["purchase_count"] = (
    articles["purchase_count"]
    .fillna(0)
)

max_pop = articles["purchase_count"].max()

if max_pop > 0:

    articles["pop_score"] = (
        articles["purchase_count"] / max_pop
    )

else:

    articles["pop_score"] = 0


# COLOR BEHAVIOR
logger.info("Building color behavior...")

user_color_pref = (
    df[df["target"] == 1]
    .merge(
        articles[
            ["article_id", "colour_group_name"]
        ],
        on="article_id",
        how="left"
    )
    .groupby("colour_group_name")
    .size()
    .reset_index(name="cnt")
)

total = user_color_pref["cnt"].sum()

user_color_pref["color_score"] = (
    user_color_pref["cnt"] / total
)

color_map = dict(zip(
    user_color_pref["colour_group_name"],
    user_color_pref["color_score"]
))

articles["color_score"] = (
    articles["colour_group_name"]
    .map(color_map)
)

articles["color_score"] = (
    articles["color_score"]
    .fillna(0)
)


# CUSTOMER HISTORY
logger.info("Building customer history...")

customer_history = (
    df[df["target"] == 1]
    .groupby("customer_id")["article_id"]
    .apply(list)
    .to_dict()
)


# TOP POPULAR
top_pop = (
    article_pop
    .sort_values(
        "purchase_count",
        ascending=False
    )["article_id"]
    .tolist()
)


# MERGE ARTICLE FEATURES
logger.info("Merging SVD features...")

df = df.merge(
    articles[["article_id"] + svd_cols],
    on="article_id",
    how="left"
)


# FEATURES
feature_cols = [
    "recency_days",
    "purchase_frequency",
    "unique_articles",
    "total_spent",
    "avg_purchase_value",
    "article_popularity",
    "article_avg_price",
    "log_popularity",
    "log_unique_customers",
    "favorite_index_encoded",
    "favorite_color_encoded",
    "price_diff",
    "is_expensive_for_user"
]

feature_cols += svd_cols


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
    "favorite_index_encoded",
    "favorite_color_encoded",
    "price_diff"
]

df[svd_cols] = df[svd_cols].fillna(0)

for c in zero_fill_cols:

    if c in df.columns:

        df[c] = df[c].fillna(0)

for c in median_fill_cols:

    if c in df.columns:

        df[c] = df[c].fillna(
            df[c].median()
        )

df = df.dropna(
    subset=["target"]
).copy()


# DATA
X = df[feature_cols]
y = df["target"]


# TRAIN VALID SPLIT
logger.info("Train validation split...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Validation shape:", X_valid.shape)


# EVALUATION FUNCTION
def evaluate_model(model, name):

    valid_pred = model.predict_proba(X_valid)[:, 1]

    pred_label = (
        valid_pred >= 0.4
    ).astype(int)

    pr_auc = average_precision_score(
        y_valid,
        valid_pred
    )

    precision = precision_score(
        y_valid,
        pred_label
    )

    recall = recall_score(
        y_valid,
        pred_label
    )

    f2 = fbeta_score(
        y_valid,
        pred_label,
        beta=2
    )

    roc_auc = roc_auc_score(
        y_valid,
        valid_pred
    )

    train_pred = model.predict_proba(
        X_train
    )[:, 1]

    train_auc = roc_auc_score(
        y_train,
        train_pred
    )

    train_logloss = log_loss(
        y_train,
        train_pred
    )

    valid_logloss = log_loss(
        y_valid,
        valid_pred
    )

    print("\n==============================")
    print(f"{name} PERFORMANCE")
    print("==============================")

    print(f"Train AUC      : {train_auc:.5f}")
    print(f"Validation AUC : {roc_auc:.5f}")
    print(f"Train LogLoss  : {train_logloss:.5f}")
    print(f"Valid LogLoss  : {valid_logloss:.5f}")

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F2": f2,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Train_AUC": train_auc,
        "Valid_AUC": roc_auc
    }


# MODELS
models = {

    "XGBoost": xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        gamma=0.1,
        tree_method="hist",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=1200,
        learning_rate=0.03,
        depth=8,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=50
    )
}


# TRAINING
summary_rows = []

for name, model in models.items():

    logger.info(f"Training {name}...")

    if name == "XGBoost":

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50
        )

    elif name == "LightGBM":

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(50)
            ]
        )

    elif name == "CatBoost":

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=100
        )

    metrics = evaluate_model(
        model,
        name
    )

    summary_rows.append(metrics)


# SAVE SUMMARY
summary = pd.DataFrame(summary_rows)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# FEATURE IMPORTANCE
fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": (
        models["XGBoost"]
        .feature_importances_
    )
})

fi = fi.sort_values(
    "importance",
    ascending=False
)

fi.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "xgb_feature_importance.csv"
    ),
    index=False
)


# HYBRID BUNDLE
logger.info("Building hybrid bundle...")

bundle = {

    # MODELS
    "xgb_model": models["XGBoost"],
    "lgb_model": models["LightGBM"],
    "cat_model": models["CatBoost"],

    # NLP
    "tfidf": tfidf,
    "svd": svd,
    "tfidf_matrix": tfidf_matrix,

    # INDEX
    "article_to_idx": article_to_idx,
    "idx_to_article": idx_to_article,

    # HISTORY
    "customer_history": customer_history,

    # POPULARITY
    "top_pop": top_pop,

    # ARTICLE META
    "article_meta": articles,

    # FEATURES
    "feature_cols": feature_cols,
    "svd_cols": svd_cols
}


# SAVE HYBRID MODEL
logger.info("Saving hybrid recommender...")

with open(
    os.path.join(
        MODEL_DIR,
        "hybrid_recommender.pkl"
    ),
    "wb"
) as f:

    pickle.dump(
        bundle,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

logger.info("Hybrid recommender saved")


# FINAL SUMMARY
print("\n==============================")
print("HYBRID RECOMMENDER READY")
print("==============================")

print("Models:")
print("- XGBoost")
print("- LightGBM")
print("- CatBoost")

print("\nRetrieval:")
print("- TF-IDF")
print("- SVD")
print("- Popularity")
print("- Customer history")
print("- Color behavior")

print("\nSaved:")
print("- hybrid_recommender.pkl")