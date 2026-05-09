import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity


print("Loading models...")

model = pickle.load(open("model/xgb_model.pkl","rb"))
tfidf = pickle.load(open("model/tfidf.pkl","rb"))
svd = pickle.load(open("model/svd.pkl","rb"))

print("Loading data...")

articles = pd.read_csv("articles_with_desc.csv")
train = pd.read_csv("train_data.csv")


# =========================
# ADD ARTICLE FEATURES
# =========================

article_features = train.groupby("article_id").agg({
    "article_popularity":"first",
    "article_avg_price":"first",
    "article_unique_customers":"first"
}).reset_index()

articles = articles.merge(
    article_features,
    on="article_id",
    how="left"
)


# =========================
# BUILD TEXT
# =========================

articles["text"] = (
    articles["prod_name"].fillna("") + " " +
    articles["product_type_name"].fillna("") + " " +
    articles["product_group_name"].fillna("") + " " +
    articles["colour_group_name"].fillna("") + " " +
    articles["department_name"].fillna("") + " " +
    articles["section_name"].fillna("") + " " +
    articles["garment_group_name"].fillna("") + " " +
    articles["clothes_description"].fillna("")
)

print("Building TF-IDF matrix...")

tfidf_matrix = tfidf.transform(articles["text"])


# =========================
# RECOMMEND FUNCTION
# =========================

def recommend(customer_id, query, top_k=100):

    print("Search query:", query)

    query_vec = tfidf.transform([query])

    sim = cosine_similarity(query_vec, tfidf_matrix)[0]

    articles["search_score"] = sim


    candidates = articles.sort_values(
        "search_score",
        ascending=False
    ).head(1000).copy()


    user = train[train.customer_id == customer_id].iloc[0]


    candidates["recency_days"] = user["recency_days"]
    candidates["purchase_frequency"] = user["purchase_frequency"]
    candidates["unique_articles"] = user["unique_articles"]
    candidates["total_spent"] = user["total_spent"]
    candidates["avg_purchase_value"] = user["avg_purchase_value"]

    candidates["favorite_index_encoded"] = user["favorite_index_encoded"]
    candidates["favorite_color_encoded"] = user["favorite_color_encoded"]


    candidates["price_diff"] = (
        candidates["article_avg_price"] - user["avg_purchase_value"]
    )

    candidates["is_expensive_for_user"] = (
        candidates["article_avg_price"] > user["avg_purchase_value"]
    ).astype(int)


    candidates["log_popularity"] = np.log1p(
        candidates["article_popularity"]
    )

    candidates["log_unique_customers"] = np.log1p(
        candidates["article_unique_customers"]
    )


    svd_vec = svd.transform(query_vec)

    for i in range(64):
        candidates[f"svd_{i}"] = svd_vec[0][i]


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

    for i in range(64):
        feature_cols.append(f"svd_{i}")


    print("Predicting ranking...")

    preds = model.predict_proba(
        candidates[feature_cols]
    )[:,1]

    candidates["score"] = preds


    result = candidates.sort_values(
        "score",
        ascending=False
    ).head(top_k)


    return result[[
        "article_id",
        "prod_name",
        "colour_group_name",
        "article_avg_price",
        "article_popularity",
        "score"
    ]]


# =========================
# TEST
# =========================

if __name__ == "__main__":

    customer_id = train.customer_id.iloc[0]

    query = "black dress"

    result = recommend(customer_id, query)

    print("\nTop recommendations:\n")

    print(result)