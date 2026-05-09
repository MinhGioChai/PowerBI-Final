import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# 1. PATH CONFIGURATION & DATA LOADING
PROCESSED_DIR = Path("data/processed")

def load_processed_data():
    """Load cleaned data from the preprocessing stage"""
    print("--- Loading Cleaned Data ---")
    df_trans = pd.read_parquet(PROCESSED_DIR / "transactions_cleaned.parquet")
    df_art = pd.read_parquet(PROCESSED_DIR / "articles_cleaned.parquet")
    df_cust = pd.read_parquet(PROCESSED_DIR / "customers_cleaned.parquet")
    return df_trans, df_art, df_cust

# 2. DATA SPLITTING
def preprocess_and_split(transactions, customers, articles):
    """Merge tables and perform time-series split"""
    print("--- Merging and Splitting ---")
    
    # Initial Merge
    df = transactions.merge(customers, on='customer_id', how='left')
    df = df.merge(articles, on='article_id', how='left')
    
    # Ensure datetime format and sort chronologically
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    df = df.sort_values('t_dat')
    
    # Time-series Split: Use the last 30 days as the Test set
    max_date = df['t_dat'].max()
    split_date = max_date - pd.Timedelta(days=30)
    
    df_train = df[df['t_dat'] < split_date].copy()
    df_test = df[df['t_dat'] >= split_date].copy()

    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    return df_train, df_test

# 3. CATEGORICAL ENCODING
def apply_categorical_encoding(df_train, df_test):
    """Encode categorical features to prevent data leakage"""
    print("--- Encoding Categorical Features ---")
    categorical_cols = ['product_type_name', 'colour_group_name', 'index_group_name', 'section_name', 'garment_group_name']
    
    for col in categorical_cols:
        # Fit ONLY on Training set to avoid leakage
        mapping = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        
        df_train[col + "_encoded"] = df_train[col].astype(str).map(mapping).fillna(-1).astype(int)
        df_test[col + "_encoded"] = df_test[col].astype(str).map(mapping).fillna(-1).astype(int)
    
    return df_train, df_test

# 4. FEATURE EXTRACTION (RFM, ITEM STATS, PREFS)
def create_rfm_features(df_train):
    """Generate and Scale Recency, Frequency, and Monetary features"""
    print("--- Creating Scaled RFM Features ---")
    max_date = df_train['t_dat'].max()
    
    # Recency
    recency = df_train.groupby('customer_id')['t_dat'].max().reset_index()
    recency['recency_days'] = (max_date - recency['t_dat']).dt.days
    
    # Frequency
    frequency = df_train.groupby('customer_id').agg(
        purchase_frequency=('customer_id', 'size'),
        unique_articles=('article_id', 'nunique')
    ).reset_index()
    
    # Monetary
    monetary = df_train.groupby('customer_id').agg(
        total_spent=('price', 'sum'),
        avg_purchase_value=('price', 'mean')
    ).reset_index()
    
    rfm = recency[['customer_id', 'recency_days']].merge(frequency, on='customer_id').merge(monetary, on='customer_id')
    
    # Feature Scaling (Crucial for most ML models)
    scale_cols = ['recency_days', 'purchase_frequency', 'unique_articles', 'total_spent', 'avg_purchase_value']
    scaler = StandardScaler()
    rfm[scale_cols] = scaler.fit_transform(rfm[scale_cols])
    
    return rfm

def create_article_features(df_train):
    """Generate Popularity, Price Stats, and Log-transformed features for items"""
    print("--- Creating Article Stats & Log Features ---")
    popularity = df_train.groupby('article_id').size().reset_index(name='article_popularity')
    
    price_stats = df_train.groupby('article_id')['price'].agg(['mean', 'std', 'min', 'max', 'nunique']).reset_index()
    price_stats.columns = ['article_id', 'article_avg_price', 'article_price_std', 'article_min_price', 'article_max_price', 'article_price_nunique']
    price_stats['article_price_std'] = price_stats['article_price_std'].fillna(0)
    
    unique_cust = df_train.groupby('article_id')['customer_id'].nunique().reset_index(name='article_unique_customers')
    
    art_feat = popularity.merge(price_stats, on='article_id').merge(unique_cust, on='article_id')
    
    # Log transformation to handle skewed distributions
    art_feat['log_popularity'] = np.log1p(art_feat['article_popularity'])
    art_feat['log_unique_customers'] = np.log1p(art_feat['article_unique_customers'])
    
    return art_feat

def create_customer_preferences(df_train):
    """Identify favorite categories for each customer using 'Mode'"""
    print("--- Creating Customer Preferences (Mode) ---")
    def most_freq(series): return series.value_counts().index[0]
    
    customer_prefs = df_train.groupby('customer_id').agg({
        'index_group_name': most_freq,
        'index_group_name_encoded': most_freq,
        'colour_group_name': most_freq,
        'colour_group_name_encoded': most_freq,
        'garment_group_name': most_freq
    }).reset_index()
    
    customer_prefs.columns = ['customer_id', 'favorite_index_group', 'favorite_index_encoded', 'favorite_color', 'favorite_color_encoded', 'favorite_garment']
    return customer_prefs

# 5. NLP PIPELINE (PRODUCT DESCRIPTIONS)
def build_article_descriptions(articles):
    """Build enriched text descriptions for Content-Based Filtering"""
    print("--- Building NLP Text Descriptions ---")
    synonyms = {'baby': ['kids', 'children'], 'kids': ['baby', 'children'], 'women': ['ladies', 'female'], 'men': ['male'], 'divided': ['unisex']}
    basic_stopwords = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for'}

    def build_description(row):
        # Repeat product_type 3x to boost its weight in TF-IDF
        parts = [row['product_type_name']] * 3 + [row['product_group_name'], row['graphical_appearance_name'], 
                 row['perceived_colour_value_name'], row['perceived_colour_master_name'], 
                 row['index_group_name'], row['section_name'], row['garment_group_name']]
        
        text = ' '.join(map(str, parts)).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Remove stopwords & Apply synonym expansion
        words = [w for w in text.split() if w not in basic_stopwords]
        expanded = words.copy()
        for w in words:
            if w in synonyms: expanded.extend(synonyms[w])
        return ' '.join(expanded)

    articles['clothes_description'] = articles.apply(build_description, axis=1)
    return articles[['article_id', 'clothes_description']]

# 6. MAIN ORCHESTRATOR
def run_feature_engineering():
    # 1. Load Data
    trans, articles, customers = load_processed_data()
    
    # 2. Cleaning & Temporal Split
    df_train, df_test = preprocess_and_split(trans, customers, articles)
    
    # 3. Categorical Encoding
    df_train, df_test = apply_categorical_encoding(df_train, df_test)
    
    # 4. Feature Engineering (Derived from Training set only)
    rfm = create_rfm_features(df_train)
    art_stats = create_article_features(df_train)
    cust_prefs = create_customer_preferences(df_train)
    art_text = build_article_descriptions(articles)
    
    # 5. Final Assembly
    print("--- Final Merging ---")
    # Merge features into Training set
    df_train = df_train.merge(rfm, on='customer_id', how='left')
    df_train = df_train.merge(art_stats, on='article_id', how='left')
    df_train = df_train.merge(cust_prefs, on='customer_id', how='left')
    df_train = df_train.merge(art_text, on='article_id', how='left')
    
    # Merge features into Test set (for model inference)
    df_test = df_test.merge(rfm, on='customer_id', how='left')
    df_test = df_test.merge(art_stats, on='article_id', how='left')
    df_test = df_test.merge(cust_prefs, on='customer_id', how='left')
    df_test = df_test.merge(art_text, on='article_id', how='left')
    
    # 6. Export to Parquet (Optimized for performance)
    df_train.to_parquet(PROCESSED_DIR / "train_data.parquet", index=False)
    df_test.to_parquet(PROCESSED_DIR / "test_data.parquet", index=False)
    print(f"Successfully saved final datasets to {PROCESSED_DIR}")

if __name__ == "__main__":
    run_feature_engineering()
