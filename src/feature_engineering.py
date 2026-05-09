import pandas as pd
import re
from pathlib import Path
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download NLP requirements
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

# 1. PATH CONFIGURATION
PROCESSED_DIR = Path("data/processed")

def load_processed_data():
    """Load cleaned data from the preprocessing stage"""
    print("--- Loading Cleaned Data ---")
    df_trans = pd.read_parquet(PROCESSED_DIR / "transactions_cleaned.parquet")
    df_art = pd.read_parquet(PROCESSED_DIR / "articles_cleaned.parquet")
    df_cust = pd.read_parquet(PROCESSED_DIR / "customers_cleaned.parquet")
    return df_trans, df_art, df_cust

def merge_and_split(transactions, customers, articles):
    """Merge tables and perform Time-series Train/Test split"""
    print("--- Merging and Splitting Data ---")
    
    # Initial Merge
    df = transactions.merge(customers, on='customer_id', how='left')
    df = df.merge(articles, on='article_id', how='left')

    # Drop columns not required for further analysis
    drop_cols = [
        'sales_channel_id', 'FN', 'Active', 'club_member_status', 'fashion_news_frequency',
        'product_code', 'product_type_no', 'graphical_appearance_no', 'colour_group_code', 
        'perceived_colour_value_id', 'perceived_colour_master_id', 'department_no', 
        'index_code', 'index_group_no', 'section_no', 'garment_group_no', 'index_name'
    ]
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Ensure t_dat is datetime
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    max_date = df['t_dat'].max()
    
    # Train-Test Split: Use the last 30 days as the Test set
    split_date = max_date - pd.Timedelta(days=30)
    df_train = df[df['t_dat'] < split_date].copy()
    df_test = df[df['t_dat'] >= split_date].copy()
    
    print(f"Data range: {df['t_dat'].min()} to {max_date}")
    print(f"Split Date: {split_date}")
    print(f"Training samples: {len(df_train)} | Test samples: {len(df_test)}")
    
    return df_train, df_test

def create_customer_rfm_features(df):
    """Calculate RFM (Recency, Frequency, Monetary) features for customers"""
    print("--- Creating Customer RFM Features ---")
    max_date = df['t_dat'].max()

    customer_features = df.groupby('customer_id').agg(
        last_purchase=('t_dat', 'max'),
        purchase_frequency=('article_id', 'count'),
        total_spent=('price', 'sum'),
        avg_purchase_value=('price', 'mean')
    ).reset_index()

    # Recency: Days since last purchase
    customer_features['recency_days'] = (max_date - customer_features['last_purchase']).dt.days
    customer_features.drop(columns=['last_purchase'], inplace=True)
    
    return customer_features

def create_article_features(df):
    """Calculate popularity and price features for articles"""
    print("--- Creating Article Popularity Features ---")
    article_features = df.groupby('article_id').agg(
        article_popularity=('customer_id', 'count'),
        article_avg_price=('price', 'mean'),
        article_price_std=('price', 'std'),
        article_unique_customers=('customer_id', 'nunique')
    ).reset_index()
    
    article_features['article_price_std'] = article_features['article_price_std'].fillna(0)
    return article_features

def create_customer_preferences(df):
    """Extract customer purchase preferences (Favorite categories)"""
    print("--- Creating Customer Preference Features ---")
    categorical_cols = ['product_type_name', 'colour_group_name', 'index_group_name', 'section_name', 'garment_group_name']
    
    # Mode calculation (most frequent purchase attributes)
    customer_prefs = df.groupby('customer_id')[categorical_cols].agg(lambda x: x.mode()[0]).reset_index()
    
    # Renaming for clarity
    customer_prefs.columns = ['customer_id', 'fav_product_type', 'fav_color', 'fav_index_group', 'fav_section', 'fav_garment']
    
    return customer_prefs

def process_text_descriptions(articles):
    """Natural Language Processing for product descriptions"""
    print("--- Processing Text Descriptions ---")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    synonyms = {
        'baby': ['kids', 'children'], 'kids': ['baby', 'children'], 'children': ['baby', 'kids'],
        'women': ['ladies', 'female'], 'ladies': ['women', 'female'],
        'men': ['male'], 'male': ['men'], 'divided': ['unisex']
    }

    def build_and_clean_text(row):
        # Boost product type importance by repeating it
        p_type = str(row['product_type_name']).lower()
        attributes = [
            p_type, p_type, p_type,
            str(row['product_group_name']), 
            str(row['graphical_appearance_name']),
            str(row['perceived_colour_value_name']), 
            str(row['perceived_colour_master_name']),
            str(row['index_group_name']), 
            str(row['section_name']), 
            str(row['garment_group_name'])
        ]
        
        text = ' '.join(attributes).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # Lemmatization and Stopword removal
        words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
        
        # Synonym expansion
        expanded = list(words)
        for w in words:
            if w in synonyms:
                expanded.extend(synonyms[w])
        
        return ' '.join(expanded)

    articles_text = articles.copy()
    articles_text['clothes_description'] = articles_text.apply(build_and_clean_text, axis=1)
    return articles_text[['article_id', 'clothes_description']]

def run_feature_engineering():
    """Main Orchestrator for Feature Engineering"""
    # 1. Load data
    transactions, articles, customers = load_processed_data()

    # 2. Merge and Split
    df_train, df_test = merge_and_split(transactions, customers, articles)

    # 3. Calculate features using df_train (to avoid Data Leakage)
    cust_rfm = create_customer_rfm_features(df_train)
    art_stats = create_article_features(df_train)
    cust_prefs = create_customer_preferences(df_train)
    art_text = process_text_descriptions(articles)

    # 4. Final Merging
    print("--- Finalizing Dataset ---")
    df_final = df_train.merge(cust_rfm, on='customer_id', how='left')
    df_final = df_final.merge(art_stats, on='article_id', how='left')
    df_final = df_final.merge(cust_prefs, on='customer_id', how='left')
    df_final = df_final.merge(art_text, on='article_id', how='left')

    # 5. Save
    train_path = PROCESSED_DIR / "train_data.parquet"
    test_path = PROCESSED_DIR / "test_data.parquet"
    
    df_final.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)

    return df_final, df_test

if __name__ == "__main__":
    run_feature_engineering()
