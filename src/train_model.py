import pandas as pd
import numpy as np
import pickle
import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "train_data.csv"
ARTICLES_PATH = "articles_with_desc.csv"

MODEL_PATH = "model/xgb_model.pkl"
TFIDF_PATH = "model/tfidf.pkl"
SVD_PATH = "model/svd.pkl"


# -------------------------
# LOAD DATA
# -------------------------

logger.info("Loading dataset...")

df = pd.read_csv(DATA_PATH)
articles = pd.read_csv(ARTICLES_PATH)

print("Train shape:", df.shape)
print("Articles shape:", articles.shape)


# -------------------------
# BUILD TEXT
# -------------------------

logger.info("Building article text...")

articles["text"] = (
    articles["prod_name"].fillna("") + " " +
    articles["product_type_name"].fillna("") + " " +
    articles["colour_group_name"].fillna("") + " " +
    articles["section_name"].fillna("") + " " +
    articles["garment_group_name"].fillna("") + " " +
    articles["clothes_description"].fillna("")
)


# -------------------------
# TF-IDF
# -------------------------

logger.info("Training TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(articles["text"])


# -------------------------
# SVD
# -------------------------

logger.info("Running SVD...")

svd = TruncatedSVD(n_components=64)

svd_matrix = svd.fit_transform(tfidf_matrix)

svd_df = pd.DataFrame(
    svd_matrix,
    columns=[f"svd_{i}" for i in range(64)]
)

articles = pd.concat([articles, svd_df], axis=1)


# -------------------------
# MERGE
# -------------------------

df = df.merge(
    articles[["article_id"] + [f"svd_{i}" for i in range(64)]],
    on="article_id",
    how="left"
)


# -------------------------
# FEATURES
# -------------------------

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

feature_cols += [f"svd_{i}" for i in range(64)]

X = df[feature_cols]
y = df["target"]


# -------------------------
# TRAIN TEST SPLIT
# -------------------------

logger.info("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# -------------------------
# TRAIN XGBOOST
# -------------------------

logger.info("Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="logloss"
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)


# -------------------------
# EVALUATION
# -------------------------

pred = model.predict_proba(X_test)[:,1]

auc = roc_auc_score(y_test, pred)

print("Validation AUC:", auc)


# -------------------------
# SAVE
# -------------------------

logger.info("Saving models...")

import os
os.makedirs("model", exist_ok=True)

pickle.dump(model, open(MODEL_PATH,"wb"))
pickle.dump(tfidf, open(TFIDF_PATH,"wb"))
pickle.dump(svd, open(SVD_PATH,"wb"))

logger.info("Training finished")