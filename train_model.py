import pandas as pd
import numpy as np
import pickle
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "train_data.csv"
ARTICLES_PATH = "articles_with_desc.csv"

os.makedirs("model", exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================
logger.info("Loading dataset...")

df = pd.read_csv(DATA_PATH)
articles = pd.read_csv(ARTICLES_PATH)

print("Train shape:", df.shape)
print("Articles shape:", articles.shape)

# =========================================================
# TEXT FEATURE
# =========================================================
logger.info("Building article text...")

articles["text"] = (
    articles["prod_name"].fillna("") + " " +
    articles["product_type_name"].fillna("") + " " +
    articles["colour_group_name"].fillna("") + " " +
    articles["section_name"].fillna("") + " " +
    articles["garment_group_name"].fillna("") + " " +
    articles["clothes_description"].fillna("")
)

# =========================================================
# TF-IDF
# =========================================================
logger.info("Training TF-IDF...")

tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english"
)

tfidf_matrix = tfidf.fit_transform(articles["text"])

# =========================================================
# SVD
# =========================================================
logger.info("Running SVD...")

svd = TruncatedSVD(n_components=64, random_state=42)
svd_matrix = svd.fit_transform(tfidf_matrix)

svd_df = pd.DataFrame(
    svd_matrix,
    columns=[f"svd_{i}" for i in range(64)]
)

articles = pd.concat([articles, svd_df], axis=1)

# =========================================================
# MERGE
# =========================================================
df = df.merge(
    articles[["article_id"] + [f"svd_{i}" for i in range(64)]],
    on="article_id",
    how="left"
)

# =========================================================
# FEATURES
# =========================================================
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

# =========================================================
# SPLIT
# =========================================================
logger.info("Splitting dataset...")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# EVALUATION FUNCTION
# =========================================================
def evaluate(model, name):
    train_pred = model.predict_proba(X_train)[:, 1]
    valid_pred = model.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    valid_auc = roc_auc_score(y_valid, valid_pred)

    train_logloss = log_loss(y_train, train_pred)
    valid_logloss = log_loss(y_valid, valid_pred)

    print("\n==============================")
    print(f"{name} PERFORMANCE")
    print("==============================")

    print(f"Train AUC    : {train_auc:.5f}")
    print(f"Valid AUC    : {valid_auc:.5f}")
    print(f"Train LogLoss: {train_logloss:.5f}")
    print(f"Valid LogLoss: {valid_logloss:.5f}")

    gap = train_auc - valid_auc

    print("\n==============================")
    print("OVERFITTING CHECK")
    print("==============================")

    print(f"AUC GAP (train - valid): {gap:.5f}")

    if gap < 0.01:
        print("✅ Good generalization")
    elif gap < 0.05:
        print("⚠️ Slight overfitting")
    else:
        print("❌ Overfitting")

    return valid_auc


# =========================================================
# XGBOOST
# =========================================================
logger.info("Training XGBoost...")

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric=["logloss", "auc"],
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    verbose=50
)

# =========================================================
# LIGHTGBM
# =========================================================
logger.info("Training LightGBM...")

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)]
)

# =========================================================
# CATBOOST
# =========================================================
logger.info("Training CatBoost...")

cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=50
)

cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid)
)

# =========================================================
# FEATURE IMPORTANCE (XGBOOST)
# =========================================================
def print_feature_importance(model):
    importance = model.get_booster().get_score(importance_type="gain")
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\n==============================")
    print("TOP FEATURE IMPORTANCE")
    print("==============================")

    for i, (feat, score) in enumerate(importance[:20]):
        print(f"{i+1}. {feat}: {score:.4f}")


# =========================================================
# RUN EVALUATION
# =========================================================
xgb_score = evaluate(xgb_model, "XGBoost")
lgb_score = evaluate(lgb_model, "LightGBM")
cat_score = evaluate(cat_model, "CatBoost")

print_feature_importance(xgb_model)

# =========================================================
# COMPARE MODELS
# =========================================================
scores = {
    "XGBoost": xgb_score,
    "LightGBM": lgb_score,
    "CatBoost": cat_score
}

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

for k, v in scores.items():
    print(f"{k}: {v:.5f}")

best_model = max(scores, key=scores.get)

print("\n==============================")
print("BEST MODEL SELECTED")
print("==============================")
print(f"Model: {best_model}")
print(f"AUC  : {scores[best_model]:.5f}")

# =========================================================
# SAVE MODELS
# =========================================================
logger.info("Saving models...")

pickle.dump(xgb_model, open("model/xgb_model.pkl", "wb"))
pickle.dump(lgb_model, open("model/lgb_model.pkl", "wb"))
pickle.dump(cat_model, open("model/cat_model.pkl", "wb"))

pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(svd, open("model/svd.pkl", "wb"))

logger.info("Training finished")