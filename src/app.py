from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from recommender import Recommender

app = Flask(__name__)

print("Loading recommender system...")
rec = Recommender()

train = pd.read_csv("train_data.csv")
articles = pd.read_csv("articles_with_desc.csv")


# ==============================
# LOGIN
# ==============================

@app.route("/", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        customer_id = request.form["customer_id"]

        return redirect(url_for("home", customer_id=customer_id))

    return render_template("login.html")


# ==============================
# HOME
# ==============================

@app.route("/home/<customer_id>")
def home(customer_id):

    user_history = train[train.customer_id == customer_id]

    # sản phẩm mua gần đây
    recent_ids = user_history.sort_values(
        "recency_days"
    ).head(10)["article_id"]

    recent_products = articles[
        articles.article_id.isin(recent_ids)
    ]

    # trending products
    trending_ids = (
        train.groupby("article_id")
        .size()
        .sort_values(ascending=False)
        .head(20)
        .index
    )

    trending_products = articles[
        articles.article_id.isin(trending_ids)
    ]

    return render_template(
        "home.html",
        customer_id=customer_id,
        recent=recent_products.to_dict("records"),
        trending=trending_products.to_dict("records")
    )


# ==============================
# SEARCH + RECOMMEND
# ==============================

@app.route("/search/<customer_id>", methods=["GET","POST"])
def search(customer_id):

    products = None
    query = ""

    if request.method == "POST":

        query = request.form["query"]

        result = rec.recommend(customer_id, query)

        products = result.to_dict("records")

    return render_template(
        "search.html",
        customer_id=customer_id,
        products=products,
        query=query
    )


# ==============================

if __name__ == "__main__":
    app.run(debug=True, port=5000)