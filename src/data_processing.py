import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path("data")

def load_raw_splits():
    print("--- Loading Raw Transaction Splits ---")
    train = pd.read_csv(BASE_DIR / "train_split.parquet")
    test = pd.read_parquet(BASE_DIR / "test_split.parquet")
    
    articles = pd.read_csv(BASE_DIR / "articles.csv")
    customers = pd.read_csv(BASE_DIR / "customers.csv")
    return train, test, articles, customers

def process_customers(customers):
    print("--- Processing Customers ---")
    customers['age'] = customers['age'].fillna(customers['age'].median())
    customers[['FN', 'Active']] = customers[['FN', 'Active']].fillna(0)
    customers['club_member_status'] = customers['club_member_status'].fillna('UNKNOWN')
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')
    customers.drop(columns=['postal_code'], inplace=True, errors='ignore')
    
    cat_cols = ['club_member_status', 'fashion_news_frequency']
    for col in cat_cols:
        customers[col] = customers[col].astype('category')
    return customers

def process_articles(articles):
    print("--- Processing Articles ---")
    
    cat_cols = [
        'product_type_name', 'product_group_name', 'graphical_appearance_name',
        'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
        'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name'
    ]
    for col in cat_cols:
        if col in articles.columns:
            articles[col] = articles[col].astype('category')
    return articles

def process_transactions(df, price_stats=None):
    df['price'] = df['price'] * 1000
    
    if price_stats is None:
        q1 = df['price'].quantile(0.25)
        q3 = df['price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        stats = {'lower': lower_bound, 'upper': upper_bound}
    else:
        stats = price_stats
        lower_bound, upper_bound = stats['lower'], stats['upper']
    
    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)].copy()
    
    return df, stats

def run_preprocess():
    train_raw, test_raw, df_art, df_cust = load_raw_splits()
    
    df_cust = process_customers(df_cust)
    df_art = process_articles(df_art)
    
    print("\n--- Cleaning Train Transactions ---")
    df_train, train_price_stats = process_transactions(train_raw)
    
    print("--- Cleaning Test Transactions (Using Train Stats) ---")
    df_test, _ = process_transactions(test_raw, price_stats=train_price_stats)
    
    print("\n--- Saving Cleaned Data ---")
    df_train.to_cvs(BASE_DIR / "train_clean.csv", index=False)
    df_test.to_csv(BASE_DIR / "test_cleaned.csv", index=False)
    
    df_art.to_csv(BASE_DIR / "articles_cleaned.csv", index=False)
    df_cust.to_csv(BASE_DIR / "customers_cleaned.csv", index=False)
    
    print(f"All splits processed and saved to: {BASE_DIR}")

if __name__ == "__main__":
    run_preprocess()