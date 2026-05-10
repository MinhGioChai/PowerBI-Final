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

# =========================================================
# CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TEST_PATH = "test_data.csv"
ARTICLES_PATH = "articles_with_desc.csv"

MODEL_DIR = "model"
OUTPUT_DIR = "outputs/test"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
logger.info("Loading test data...")

test = pd.read_csv(TEST_PATH)
articles = pd.read_csv(ARTICLES_PATH)

print("Test shape:", test.shape)
print("Articles shape:", articles.shape)

if "target" not in test.columns:
    raise ValueError("test_data.csv must contain target column")


# =========================================================
# LOAD ARTIFACTS
# =========================================================
logger.info("Loading model artifacts...")

model = pickle.load(
    open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb")
)

tfidf = pickle.load(
    open(os.path.join(MODEL_DIR, "tfidf.pkl"), "rb")
)

svd = pickle.load(
    open(os.path.join(MODEL_DIR, "svd.pkl"), "rb")
)


# =========================================================
# BUILD ARTICLE TEXT
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
# TFIDF + SVD
# =========================================================
logger.info("Transforming article text...")

tfidf_matrix = tfidf.transform(articles["text"])
svd_matrix = svd.transform(tfidf_matrix)

svd_cols = [f"svd_{i}" for i in range(svd_matrix.shape[1])]

svd_df = pd.DataFrame(
    svd_matrix,
    columns=svd_cols
)

articles = pd.concat(
    [articles.reset_index(drop=True), svd_df],
    axis=1
)


# =========================================================
# MERGE ARTICLE FEATURES
# =========================================================
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
print("Rows before merge:", rows_before_merge)
print("Rows after merge :", len(test))
print("Missing SVD rows :", test[svd_cols].isna().all(axis=1).mean())


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

feature_cols += svd_cols

missing_cols = [c for c in feature_cols if c not in test.columns]

if len(missing_cols) > 0:
    raise ValueError(f"Missing feature columns: {missing_cols}")


# =========================================================
# HANDLE MISSING
# =========================================================
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

# SVD
test[svd_cols] = test[svd_cols].fillna(0)

# zero fill
for c in zero_fill_cols:
    if c in test.columns:
        test[c] = test[c].fillna(0)

# median fill
for c in median_fill_cols:
    if c in test.columns:
        test[c] = test[c].fillna(test[c].median())

# only target
test = test.dropna(subset=["target"]).copy()

print("\n==============================")
print("MISSING AFTER FILL")
print("==============================")
print("Remaining missing:", test[feature_cols].isna().sum().sum())
print("Final test shape:", test.shape)


# =========================================================
# TEST DATA
# =========================================================
X_test = test[feature_cols]
y_test = test["target"]

print("\nTarget rate:", y_test.mean())


# =========================================================
# PREDICT
# =========================================================
logger.info("Running XGBoost test...")

preds = model.predict_proba(X_test)[:, 1]
pred_label = (preds >= 0.5).astype(int)


# =========================================================
# METRICS
# =========================================================
pr_auc = average_precision_score(y_test, preds)
roc_auc = roc_auc_score(y_test, preds)
ll = log_loss(y_test, preds)

precision = precision_score(y_test, pred_label)
recall = recall_score(y_test, pred_label)
f2 = fbeta_score(y_test, pred_label, beta=2)

print("\n==============================")
print("XGBOOST TEST")
print("==============================")
print(f"PR AUC    : {pr_auc:.5f}")
print(f"ROC AUC   : {roc_auc:.5f}")
print(f"LogLoss   : {ll:.5f}")
print(f"Precision : {precision:.5f}")
print(f"Recall    : {recall:.5f}")
print(f"F2        : {f2:.5f}")


# =========================================================
# SUMMARY
# =========================================================
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


# =========================================================
# PREDICTIONS
# =========================================================
predictions = test[
    ["customer_id", "article_id", "target"]
].copy()

predictions["predicted_probability"] = preds
predictions["predicted_label"] = pred_label
predictions["Model"] = "XGBoost"


# =========================================================
# CALIBRATION
# =========================================================
bins = np.linspace(0, 1, 11)

cal = predictions[
    ["target", "predicted_probability"]
].copy()

cal["bin"] = pd.cut(
    cal["predicted_probability"],
    bins=bins,
    include_lowest=True
)

calibration = cal.groupby("bin").agg(
    observed_rate=("target", "mean"),
    avg_pred=("predicted_probability", "mean"),
    n=("target", "count")
).reset_index()

calibration["Model"] = "XGBoost"


# =========================================================
# RISK BANDS
# =========================================================
risk = predictions.copy()

risk["risk_band"] = pd.qcut(
    risk["predicted_probability"],
    q=4,
    labels=["Low", "Medium", "High", "Critical"],
    duplicates="drop"
)

risk_bands = risk.groupby("risk_band").agg(
    count=("target", "count"),
    actual_rate=("target", "mean"),
    avg_pred=("predicted_probability", "mean")
).reset_index()

risk_bands["Model"] = "XGBoost"


# =========================================================
# CONFUSION MATRIX
# =========================================================
tn, fp, fn, tp = confusion_matrix(
    y_test,
    pred_label
).ravel()

confusion = pd.DataFrame([{
    "Model": "XGBoost",
    "True Positive": tp,
    "True Negative": tn,
    "False Positive": fp,
    "False Negative": fn
}])


# =========================================================
# FEATURE IMPORTANCE
# =========================================================
feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values(
    "importance",
    ascending=False
)

feature_importance["Model"] = "XGBoost"


# =========================================================
# TOP-10 RECOMMENDATIONS
# =========================================================
topk = predictions.copy()

topk["rank"] = (
    topk.groupby("customer_id")["predicted_probability"]
    .rank(method="first", ascending=False)
)

topk = topk[topk["rank"] <= 10].copy()

topk = topk.sort_values(
    ["customer_id", "rank"]
)


# =========================================================
# SAVE
# =========================================================
summary.to_csv(
    os.path.join(OUTPUT_DIR, "test_model_summary.csv"),
    index=False
)

predictions.to_csv(
    os.path.join(OUTPUT_DIR, "test_predictions_vs_actual.csv"),
    index=False
)

calibration.to_csv(
    os.path.join(OUTPUT_DIR, "test_calibration_curve.csv"),
    index=False
)

risk_bands.to_csv(
    os.path.join(OUTPUT_DIR, "test_risk_bands.csv"),
    index=False
)

confusion.to_csv(
    os.path.join(OUTPUT_DIR, "test_confusion_matrix.csv"),
    index=False
)

feature_importance.to_csv(
    os.path.join(OUTPUT_DIR, "test_feature_importance.csv"),
    index=False
)

topk.to_csv(
    os.path.join(OUTPUT_DIR, "top10_recommendations.csv"),
    index=False
)

logger.info("Test finished")