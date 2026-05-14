import numpy as np
import pandas as pd
import os
import re
from sklearn.preprocessing import StandardScaler

def load_data():
    data_dir = os.path.join(os.getcwd(), "data")
    print(f"--- Loading data from {data_dir} ---")
    
    df_train = pd.read_csv(os.path.join(data_dir, "train_clean.csv"))
    df_test = pd.read_csv(os.path.join(data_dir, "test_clean.csv"))
    articles = pd.read_csv(os.path.join(data_dir, "articles_clean.csv"))

    df_train['t_dat'] = pd.to_datetime(df_train['t_dat'])
    df_test['t_dat'] = pd.to_datetime(df_test['t_dat'])
    
    return df_train, df_test, articles

def apply_label_encoding(train, test, cols):
    for col in cols:
        unique_vals = train[col].astype(str).unique()
        mapping = {v: i for i, v in enumerate(unique_vals)}

        train[f"{col}_encoded"] = train[col].astype(str).map(mapping).fillna(-1).astype(int)
        test[f"{col}_encoded"] = test[col].astype(str).map(mapping).fillna(-1).astype(int)
    return train, test

def extract_rfm_features(df):
    print("--- Extracting RFM Features ---")
    df_rfm = df.sort_values('t_dat')
    max_date = df_rfm['t_dat'].max()

    # Recency
    rfm = df_rfm.groupby('customer_id')['t_dat'].max().reset_index()
    rfm.columns = ['customer_id', 'last_purchase']
    rfm['recency_days'] = (max_date - rfm['last_purchase']).dt.days
    
    # Frequency & Monetary
    stats = df_rfm.groupby('customer_id').agg(
        purchase_frequency=('customer_id', 'size'),
        unique_articles=('article_id', 'nunique'),
        total_spent=('price', 'sum'),
        avg_purchase_value=('price', 'mean')
    ).reset_index()

    rfm = rfm.merge(stats, on='customer_id').drop(columns=['last_purchase'])

    # Scaling
    scale_cols = ['recency_days', 'purchase_frequency', 'unique_articles', 'total_spent', 'avg_purchase_value']
    scaler = StandardScaler()
    rfm[scale_cols] = scaler.fit_transform(rfm[scale_cols])
    
    return rfm

def extract_article_features(df):
    print("--- Extracting Article Features ---")
    art_stats = df.groupby('article_id').agg(
        article_popularity=('customer_id', 'size'),
        article_unique_customers=('customer_id', 'nunique'),
        article_avg_price=('price', 'mean'),
        article_min_price=('price', 'min'),
        article_max_price=('price', 'max'),
        article_price_std=('price', 'std'),
        article_price_nunique=('price', 'nunique')
    ).reset_index()
    
    art_stats['article_price_std'] = art_stats['article_price_std'].fillna(0)

    # Log features
    art_stats['log_popularity'] = np.log1p(art_stats['article_popularity'])
    art_stats['log_unique_customers'] = np.log1p(art_stats['article_unique_customers'])
    
    return art_stats

def extract_preferences(df):
    def most_freq(series):
        return series.value_counts().index[0]

    prefs = df.groupby('customer_id').agg({
        'index_group_name': most_freq,
        'index_group_name_encoded': most_freq,
        'colour_group_name': most_freq,
        'colour_group_name_encoded': most_freq,
        'garment_group_name': most_freq
    }).reset_index()

    prefs.columns = [
        'customer_id', 'favorite_index_group', 'favorite_index_encoded',
        'favorite_color', 'favorite_color_encoded', 'favorite_garment'
    ]
    return prefs

def build_clean_description(articles_df):
    print("--- Processing Article Descriptions ---")
    synonyms = {
        'baby': ['kids', 'children'], 'kids': ['baby', 'children'],
        'women': ['ladies', 'female'], 'ladies': ['women', 'female'],
        'men': ['male'], 'divided': ['unisex']
    }
    stopwords = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for'}

    def process_row(row):
        fields = ['product_type_name', 'product_group_name', 'graphical_appearance_name', 
                  'perceived_colour_value_name', 'perceived_colour_master_name', 
                  'index_group_name', 'section_name', 'garment_group_name']
        text = ' '.join([str(row[f]) for f in fields]).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        words = text.split()
        clean_words = [w for w in words if w not in stopwords]
        expanded = clean_words.copy()
        for w in clean_words:
            if w in synonyms:
                expanded.extend(synonyms[w])
        return ' '.join(expanded)

    articles_df['clothes_description'] = articles_df.apply(process_row, axis=1)
    return articles_df

def assemble_dataset(base_df, rfm, art_feat, prefs, ratio=2):
    data = base_df[['customer_id', 'article_id']].drop_duplicates().copy()
    data['target'] = 1

    users = base_df['customer_id'].unique()
    items = base_df['article_id'].unique()
    pos_set = set(zip(data['customer_id'], data['article_id']))
    
    neg_samples = []
    n_neg = len(data) * ratio
    while len(neg_samples) < n_neg:
        u, i = np.random.choice(users), np.random.choice(items)
        if (u, i) not in pos_set:
            neg_samples.append((u, i))
    
    df_neg = pd.DataFrame(neg_samples, columns=['customer_id', 'article_id'])
    df_neg['target'] = 0
    
    df_final = pd.concat([data, df_neg], ignore_index=True).sample(frac=1).reset_index(drop=True)
    df_final = df_final.merge(rfm, on='customer_id', how='left') \
                       .merge(art_feat, on='article_id', how='left') \
                       .merge(prefs, on='customer_id', how='left')
    
    drop_cats = ['product_type_name', 'colour_group_name', 'index_group_name', 
                 'section_name', 'garment_group_name', 'favorite_index_group', 
                 'favorite_color', 'favorite_garment']
    df_final.drop(columns=drop_cats, inplace=True, errors='ignore')
    
    num_cols = df_final.select_dtypes(include=['number']).columns
    df_final[num_cols] = df_final[num_cols].fillna(0)
    
    for col in df_final.select_dtypes(exclude=['number']).columns:
        df_final[col] = df_final[col].astype(str).fillna("missing")

    # Cross Features
    df_final['price_diff'] = df_final['article_avg_price'] - df_final['avg_purchase_value']
    df_final['is_expensive_for_user'] = (df_final['price_diff'] > 0).astype(int)
    
    return df_final

if __name__ == "__main__":
    # Load data
    df_train, df_test, articles = load_data()

    # Categorical Encoding
    categorical_cols = ['product_type_name', 'colour_group_name', 'index_group_name', 'section_name', 'garment_group_name']
    df_train, df_test = apply_label_encoding(df_train, df_test, categorical_cols)

    # Feature Extraction
    rfm_features = extract_rfm_features(df_train)
    article_features = extract_article_features(df_train)
    customer_prefs = extract_preferences(df_train)
    articles = build_clean_description(articles)

    # Cập nhật thông tin sở thích vào base data trước khi assemble
    df_train = df_train.merge(customer_prefs, on='customer_id', how='left')
    df_test = df_test.merge(customer_prefs, on='customer_id', how='left')

    # Assemble Final Datasets
    print("--- Assembling Final Datasets ---")
    train_final = assemble_dataset(df_train, rfm_features, article_features, customer_prefs)
    test_final = assemble_dataset(df_test, rfm_features, article_features, customer_prefs)

    # Save Output
    train_final.to_csv("data/train_data.csv", index=False)
    test_final.to_csv("data/test_data.csv", index=False)
    articles[['article_id', 'clothes_description']].to_csv("data/articles_with_desc.csv", index=False)

    print("Done! Feature Engineering completed.")
