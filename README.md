# PowerBI-Final


# Model

This project implements a hybrid fashion recommendation system using:

- TF-IDF text embeddings
- Truncated SVD dimensionality reduction
- Popularity-based recommendation
- Customer behavior modeling
- Gradient boosting ranking models

The system predicts the probability that a customer will purchase a product and generates personalized Top-K recommendations.

---

# Model Pipeline

## 1. Data Loading

Input datasets:

```text
train_data.csv
test_data.csv
articles_with_desc.csv
```

The pipeline loads:

- customer behavior data
- article metadata
- transaction-derived features
- product descriptions

---

## 2. Text Feature Engineering

Product metadata is merged into a single text field:

- product name
- product type
- color
- section
- garment group
- product description

Example:

```python
articles["text"] = (
    articles["prod_name"] + " " +
    articles["product_type_name"] + " " +
    articles["colour_group_name"] + " " +
    articles["section_name"] + " " +
    articles["garment_group_name"] + " " +
    articles["clothes_description"]
)
```

---

## 3. TF-IDF Vectorization

The text is transformed into sparse TF-IDF vectors.

Purpose:

- capture semantic meaning
- measure product similarity
- support content-based recommendation

Generated artifact:

```text
tfidf_matrix
```

---

## 4. SVD (Matrix Decomposition)

TF-IDF vectors are compressed using Truncated SVD.

Purpose:

- reduce dimensionality
- remove noise
- learn latent semantic patterns

Transformation:

```text
20000 TF-IDF dimensions
        ↓
64 latent semantic features
```

Generated features:

```text
svd_0 ... svd_63
```

These features are later used as machine learning inputs.

---

# Recommendation Logic

The recommendation system works in two stages.

---

## Stage 1 — Retrieval Layer

The system first retrieves candidate products using:

### Content Similarity

Cosine similarity between products:

```text
TF-IDF + SVD
```

This retrieves visually and semantically similar items.

---

### Popularity / Trend Score

Products are ranked using:

```text
purchase_count
```

Frequently purchased products receive higher scores.

---

### Customer Behavior

Behavior features include:

- favorite colors
- purchase frequency
- spending behavior
- recency
- price preference

---

## Final Hybrid Retrieval Score

The retrieval layer combines all signals:

```text
Final Score =
0.65 × similarity
+ 0.25 × popularity
+ 0.10 × behavior
```

This ranking is used before machine learning prediction.

---

# Machine Learning Models

Three gradient boosting models are trained independently:

- XGBoost
- LightGBM
- CatBoost

Each model receives:

- customer features
- product statistics
- SVD latent vectors
- behavioral features

Target label:

```text
target = 1 → purchased
target = 0 → not purchased
```

---

# Training Flow

```text
Load datasets
    ↓
Build article text
    ↓
TF-IDF vectorization
    ↓
SVD decomposition
    ↓
Build popularity features
    ↓
Build behavior features
    ↓
Merge all features
    ↓
Train/validation split
    ↓
Train XGBoost
Train LightGBM
Train CatBoost
    ↓
Evaluate metrics
    ↓
Save hybrid recommender
```

---

# Evaluation Metrics

Models are evaluated using:

- Precision
- Recall
- F2 Score
- PR AUC
- ROC AUC
- LogLoss

---

# Training Outputs

## Main Hybrid Model

```text
model/hybrid_recommender.pkl
```

Contains:

- trained models
- TF-IDF vectorizer
- SVD model
- TF-IDF matrix
- article mappings
- customer history
- popularity data
- feature columns

---

## Model Comparison

```text
outputs/model_comparison.csv
```

Performance comparison between:

- XGBoost
- LightGBM
- CatBoost

---

## Feature Importance

```text
outputs/xgb_feature_importance.csv
```

Most important features learned by XGBoost.

---

# Testing Pipeline

The testing pipeline:

1. Loads the saved hybrid model
2. Rebuilds SVD features for test data
3. Generates prediction probabilities
4. Produces Top-K recommendations
5. Evaluates recommendation performance

---

# Testing Outputs

## Prediction Results

```text
outputs/test/test_predictions.csv
```

Contains:

- customer_id
- article_id
- predicted_probability
- predicted_label

---

## Top-K Recommendations

```text
outputs/test/topk_recommendations.csv
```

Top recommended products per customer.

---

## Feature Importance

```text
outputs/test/feature_importance.csv
```

Feature importance during testing.

---

## Confusion Matrix

```text
outputs/test/confusion_matrix.csv
```

Contains:

- True Positive
- True Negative
- False Positive
- False Negative

---

## Old Customer Recommendations

```text
outputs/test/old_customer_recommendation.csv
```

Recommendations for existing customers using:

- purchase history
- similarity
- trend score
- customer behavior

---

## New Customer Recommendations

```text
outputs/test/new_customer_recommendation.csv
```

Cold-start recommendations using:

- popularity
- similarity
- trend ranking

---

# Web Recommendation System

The Flask application:

```text
app.py
```

Provides:

- customer login
- product search
- similar item recommendation
- personalized recommendations

The web system uses:

- cosine similarity
- popularity trend
- customer behavior
- hybrid ranking score

to generate real-time recommendations.

---

# Final Recommendation Architecture

```text
Customer History
        +
Product Similarity
        +
Trend / Popularity
        +
Behavior Features
        ↓
Hybrid Retrieval Ranking
        ↓
XGBoost / LightGBM / CatBoost
        ↓
Purchase Probability
        ↓
Top-K Recommendation
```

