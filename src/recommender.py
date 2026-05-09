import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity


class Recommender:

    def __init__(self):

        print("Loading models...")

        self.model = pickle.load(open("model/xgb_model.pkl","rb"))
        self.tfidf = pickle.load(open("model/tfidf.pkl","rb"))
        self.svd = pickle.load(open("model/svd.pkl","rb"))

        print("Loading data...")

        self.articles = pd.read_csv("articles_with_desc.csv")
        self.train = pd.read_csv("train_data.csv")

        self.articles["text"] = (
            self.articles["prod_name"].fillna("") + " " +
            self.articles["clothes_description"].fillna("")
        )

        self.tfidf_matrix = self.tfidf.transform(self.articles["text"])

        print("Recommender ready")


    def recommend(self, customer_id, query, top_k=100):

        query_vec = self.tfidf.transform([query])

        sim = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        self.articles["search_score"] = sim

        candidates = self.articles.sort_values(
            "search_score",
            ascending=False
        ).head(1000)

        user = self.train[self.train.customer_id==customer_id].iloc[0]

        candidates["avg_purchase_value"] = user["avg_purchase_value"]

        candidates["price_diff"] = (
            candidates["article_avg_price"] - user["avg_purchase_value"]
        )

        feature_cols = [
            "search_score",
            "article_popularity",
            "price_diff"
        ]

        preds = self.model.predict_proba(
            candidates[feature_cols]
        )[:,1]

        candidates["score"] = preds

        return candidates.sort_values(
            "score",
            ascending=False
        ).head(top_k)