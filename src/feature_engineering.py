import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATASET
def load_data():
    data_dir = os.path.join(os.getcwd(), "data")
    train_path = os.path.join(data_dir, "train_clean.csv")
    test_path = os.path.join(data_dir, "test_clean.csv")
    articles_path = os.path.join(data_dir, "articles_clean.csv")

    print("--- Loading data from /data folder ---")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    articles = pd.read_csv(articles_path)

    df_train['t_dat'] = pd.to_datetime(df_train['t_dat'])
    df_test['t_dat'] = pd.to_datetime(df_test['t_dat'])
    
    return df_train, df_test, articles

# 2. CATEGORICAL ENCODING
def encode_features(df_train, df_test):
    print("--- Encoding Categorical Features ---")
    categorical_cols = ['product_type_name', 'colour_group_name', 'index_group_name', 'section_name', 'garment_group_name']
    
    for col in categorical_cols:
        mapping = {v: i for i, v in enumerate(df_train[col].astype(str).unique())}
        df_train[col + "_encoded"] = df_train[col].astype(str).map(mapping).fillna(-1).astype(int)
        df_test[col + "_encoded"] = df_test[col].astype(str).map(mapping).fillna(-1).astype(int)
        
    return df_train, df_test

# 3. ARTICLE METADATA & STATS
def process_articles(df_train, articles):
    print("--- Processing Article Stats and Descriptions ---")
    
    # 3.1. Basic Stats from Train
    popularity = df_train.groupby('article_id').size().reset_index(name='article_popularity')
    price_stats = df_train.groupby('article_id')['price'].agg(['mean', 'std', 'min', 'max', 'nunique']).reset_index()
    price_stats.columns = ['article_id', 'article_avg_price', 'article_price_std', 'article_min_price', 'article_max_price', 'article_price_nunique']
    
    # 3.2. NLP Pipeline
    def build_desc(row):
        parts = [row['product_type_name']] * 3 + [row['product_group_name'], row['graphical_appearance_name'], 
                 row['perceived_colour_value_name'], row['index_group_name'], row['section_name'], row['garment_group_name']]
        text = ' '.join(map(str, parts)).lower()
        return re.sub(r'[^a-z0-9\s]', ' ', text)

    articles['clothes_description'] = articles.apply(build_desc, axis=1)
    
    # Save articles_with_desc.csv
    articles[['article_id', 'clothes_description']].to_csv("data/articles_with_desc.csv", index=False)
    print("Saved: data/articles_with_desc.csv")

    # Combine all article features
    article_features = popularity.merge(price_stats, on='article_id', how='left')
    article_features['log_popularity'] = np.log1p(article_features['article_popularity'])

    return article_features

# 4. CUSTOMER PROFILING
def create_customer_features(df_train):
    print("--- Generating RFM & Customer Preferences ---")
    max_date = df_train['t_dat'].max()
    
    rfm = df_train.groupby('customer_id').agg(
        last_purchase=('t_dat', 'max'),
        purchase_frequency=('customer_id', 'size'),
        unique_articles=('article_id', 'nunique'),
        total_spent=('price', 'sum'),
        avg_purchase_value=('price', 'mean')
    ).reset_index()
    
    rfm['recency_days'] = (max_date - rfm['last_purchase']).dt.days
    rfm.drop(columns=['last_purchase'], inplace=True)
    scaler = StandardScaler()
    rfm[['recency_days', 'purchase_frequency', 'unique_articles', 'total_spent', 'avg_purchase_value']] = \
        scaler.fit_transform(rfm[['recency_days', 'purchase_frequency', 'unique_articles', 'total_spent', 'avg_purchase_value']])

    def most_freq(series):
        return series.value_counts().index[0] if not series.empty else None

    cust_prefs = df_train.groupby('customer_id').agg({
        'index_group_name_encoded': most_freq,
        'colour_group_name_encoded': most_freq,
    }).reset_index()
    cust_prefs.columns = ['customer_id', 'favorite_index_encoded', 'favorite_color_encoded']

    return rfm, cust_prefs

# 5. NEGATIVE SAMPLING & FINAL EXPORT
def build_final_set(df_pos, rfm, art_feats, cust_prefs, users, items, filename):
    print(f"--- Building {filename} ---")
    pos_set = set(zip(df_pos['customer_id'], df_pos['article_id']))
    neg_samples = []
    n_neg = len(df_pos) * 2

    # Negative Sampling
    while len(neg_samples) < n_neg:
        u, i = np.random.choice(users), np.random.choice(items)
        if (u, i) not in pos_set:
            neg_samples.append((u, i))

    df_neg = pd.DataFrame(neg_samples, columns=['customer_id', 'article_id'])
    df_neg['target'] = 0
    
    # Merge Features
    final_df = pd.concat([df_pos, df_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    final_df = final_df.merge(rfm, on='customer_id', how='left')
    final_df = final_df.merge(art_feats, on='article_id', how='left')
    final_df = final_df.merge(cust_prefs, on='customer_id', how='left')

    # Add Interaction Features
    final_df['price_diff'] = final_df['article_avg_price'] - final_df['avg_purchase_value']
    final_df['is_expensive_for_user'] = (final_df['price_diff'] > 0).astype(int)
    final_df.fillna(0, inplace=True)
    
    final_df.to_csv(f"data/{filename}", index=False)
    print(f"Saved: data/{filename} (Shape: {final_df.shape})")

# MAIN
if __name__ == "__main__":
    df_train, df_test, articles = load_data()
    df_train, df_test = encode_features(df_train, df_test)
    
    # 1. Process Articles (Popularity & Stats)
    art_feats = process_articles(df_train, articles)
    
    # 2. Process Customers (RFM & Preferences)
    rfm, cust_prefs = create_customer_features(df_train)
    
    # 3. Build Train Final
    df_pos_train = df_train[['customer_id', 'article_id']].drop_duplicates().copy()
    df_pos_train['target'] = 1
    build_final_set(df_pos_train, rfm, art_feats, cust_prefs, 
                    df_train['customer_id'].unique(), df_train['article_id'].unique(), "train_data.csv")
    
    # 4. Build Test Final
    df_pos_test = df_test[['customer_id', 'article_id']].drop_duplicates().copy()
    df_pos_test['target'] = 1
    build_final_set(df_pos_test, rfm, art_feats, cust_prefs, 
                    df_test['customer_id'].unique(), df_test['article_id'].unique(), "test_data.csv")

    print("\n--- ALL PROCESSES COMPLETED SUCCESSFULLY ---")