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


# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "train_data.csv"
ARTICLES_PATH = "articles_with_desc.csv"

os.makedirs("model", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
logger.info("Loading dataset...")

df = pd.read_csv(DATA_PATH)
articles = pd.read_csv(ARTICLES_PATH)

print("Train shape:", df.shape)
print("Articles shape:", articles.shape)


# =========================================================
# BUILD TEXT
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

svd = TruncatedSVD(
    n_components=64,
    random_state=42
)

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
    X,
    y,
    test_size=0.2,
    random_state=42
)

train_idx = X_train.index
valid_idx = X_valid.index


# =========================================================
# DATA LEAKAGE CHECK
# =========================================================
print("\n==============================")
print("DATA LEAKAGE CHECK")
print("==============================")

overlap = len(set(train_idx).intersection(set(valid_idx)))

print("Overlap rows:", overlap)

if overlap == 0:
    print("✅ No row overlap")
else:
    print("⚠️ Potential leakage")


train_customers = set(df.loc[train_idx, "customer_id"])
valid_customers = set(df.loc[valid_idx, "customer_id"])

customer_overlap = len(train_customers.intersection(valid_customers))

print("\nCustomer overlap:", customer_overlap)

train_articles = set(df.loc[train_idx, "article_id"])
valid_articles = set(df.loc[valid_idx, "article_id"])

article_overlap = len(train_articles.intersection(valid_articles))

print("Article overlap:", article_overlap)


# =========================================================
# POPULARITY BASELINE
# =========================================================
print("\n==============================")
print("POPULARITY BASELINE")
print("==============================")

baseline_pred = X_valid["article_popularity"]

baseline_pr = average_precision_score(y_valid, baseline_pred)
baseline_auc = roc_auc_score(y_valid, baseline_pred)

print(f"PR AUC  : {baseline_pr:.5f}")
print(f"ROC AUC : {baseline_auc:.5f}")


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def evaluate_model(model, name):

    valid_pred = model.predict_proba(X_valid)[:, 1]
    pred_label = (valid_pred >= 0.5).astype(int)

    pr_auc = average_precision_score(y_valid, valid_pred)
    precision = precision_score(y_valid, pred_label)
    recall = recall_score(y_valid, pred_label)
    f2 = fbeta_score(y_valid, pred_label, beta=2)
    roc_auc = roc_auc_score(y_valid, valid_pred)

    train_pred = model.predict_proba(X_train)[:, 1]

    train_auc = roc_auc_score(y_train, train_pred)
    valid_auc = roc_auc

    train_logloss = log_loss(y_train, train_pred)
    valid_logloss = log_loss(y_valid, valid_pred)

    print("\n==============================")
    print(f"{name} PERFORMANCE")
    print("==============================")

    print(f"Train AUC      : {train_auc:.5f}")
    print(f"Validation AUC : {valid_auc:.5f}")
    print(f"Train LogLoss  : {train_logloss:.5f}")
    print(f"Valid LogLoss  : {valid_logloss:.5f}")

    gap = train_auc - valid_auc

    print("\n==============================")
    print("OVERFIT CHECK")
    print("==============================")
    print(f"AUC gap: {gap:.5f}")

    if gap < 0.01:
        print("✅ Good generalization")
    elif gap < 0.05:
        print("⚠️ Slight overfitting")
    else:
        print("❌ Overfitting")

    return {
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F2": f2,
        "PR_AUC": pr_auc,
        "ROC_AUC": roc_auc,
        "Train_AUC": train_auc,
        "Valid_AUC": valid_auc
    }, valid_pred


def check_best_iteration(model, name):

    print("\n==============================")
    print(f"{name} CONVERGENCE CHECK")
    print("==============================")

    if name == "XGBoost":

        total_iter = model.n_estimators

        try:
            best_iter = model.best_iteration
        except AttributeError:
            best_iter = total_iter - 1

    elif name == "LightGBM":

        total_iter = model.n_estimators

        if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
            best_iter = model.best_iteration_
        else:
            best_iter = total_iter - 1

    elif name == "CatBoost":

        total_iter = model.get_params()["iterations"]

        try:
            best_iter = model.get_best_iteration()
            if best_iter is None:
                best_iter = total_iter - 1
        except:
            best_iter = total_iter - 1

    else:
        return None, None

    print(f"Best iteration : {best_iter}")
    print(f"Total rounds   : {total_iter}")

    if best_iter >= total_iter - 1:
        print("⚠️ Best iteration at last round → model may not have converged")
    elif best_iter >= total_iter * 0.9:
        print("⚠️ Best iteration near end")
    else:
        print("✅ Converged reasonably well")

    return best_iter, total_iter


# =========================================================
# MODELS
# =========================================================
models = {
    "XGBoost": xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1
    ),

    "LightGBM": lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1
    ),

    "CatBoost": CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=50
    )
}


# =========================================================
# TRAIN
# =========================================================
summary_rows = []
curve_rows = {}

for name, model in models.items():

    logger.info(f"Training {name}...")

    if name == "XGBoost":

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=50
        )

    elif name == "CatBoost":

        model.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid)
        )

    else:

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)]
        )

    check_best_iteration(model, name)

    metrics, valid_pred = evaluate_model(model, name)

    summary_rows.append(metrics)

    tmp = pd.DataFrame({
        "y_true": y_valid,
        "pred": valid_pred
    })

    bins = np.linspace(0, 1, 11)

    tmp["bin"] = pd.cut(
        tmp["pred"],
        bins=bins,
        include_lowest=True
    )

    grouped = tmp.groupby("bin").agg(
        observed_rate=("y_true", "mean"),
        avg_pred=("pred", "mean")
    ).reset_index()

    grouped["Model"] = name

    curve_rows[name] = grouped


# =========================================================
# SAVE MODEL COMPARISON
# =========================================================
summary = pd.DataFrame(summary_rows)

summary.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# =========================================================
# SAVE VALIDATION CURVE
# =========================================================
validation_curve = pd.concat(
    curve_rows.values(),
    ignore_index=True
)

validation_curve.to_csv(
    "outputs/validation_pr_auc_by_model.csv",
    index=False
)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
xgb_model = models["XGBoost"]

fi = pd.DataFrame({
    "feature": feature_cols,
    "importance": xgb_model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

fi.to_csv(
    "outputs/xgb_feature_importance.csv",
    index=False
)


# =========================================================
# SAVE MODELS
# =========================================================
pickle.dump(
    models["XGBoost"],
    open("model/xgb_model.pkl", "wb")
)

pickle.dump(
    models["LightGBM"],
    open("model/lgb_model.pkl", "wb")
)

pickle.dump(
    models["CatBoost"],
    open("model/cat_model.pkl", "wb")
)

pickle.dump(
    tfidf,
    open("model/tfidf.pkl", "wb")
)

pickle.dump(
    svd,
    open("model/svd.pkl", "wb")
)

logger.info("Training finished")