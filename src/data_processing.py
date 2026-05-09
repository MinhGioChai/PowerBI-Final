import pandas as pd
from pathlib import Path

# 1. PATH CONFIGURATION
DATA_PATH = "data/"
BASE_DIR = Path("data") 
PROCESSED_PATH = BASE_DIR / "processed"
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

TRANSACTIONS_PATH = f"{DATA_PATH}transactions_6_months.csv"
ARTICLES_PATH = f"{DATA_PATH}articles.csv"
CUSTOMERS_PATH = f"{DATA_PATH}customers.csv"

def load_data():
    """Load raw data from CSV files"""
    print("--- Loading Data ---")
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    articles = pd.read_csv(ARTICLES_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)
    return transactions, articles, customers

def process_customers(customers):
    """Handle missing values and cast types for the Customers table"""
    print("--- Processing Customers ---")
    
    # Fill missing values
    customers['age'] = customers['age'].fillna(customers['age'].median())
    customers[['FN', 'Active']] = customers[['FN', 'Active']].fillna(0)
    customers['club_member_status'] = customers['club_member_status'].fillna('UNKNOWN')
    customers['fashion_news_frequency'] = customers['fashion_news_frequency'].fillna('NONE')
    
    # Drop unnecessary columns
    customers.drop(columns=['postal_code'], inplace=True, errors='ignore')
    
    # Cast to Category to save RAM
    cat_cols = ['club_member_status', 'fashion_news_frequency']
    for col in cat_cols:
        customers[col] = customers[col].astype('category')
        
    return customers

def process_articles(articles):
    """Clean and optimize the Articles table"""
    print("--- Processing Articles ---")
    
    # Drop long text descriptions to save memory
    articles.drop(columns=['detail_desc'], inplace=True, errors='ignore')
    
    # Cast categorical columns to category type
    cat_cols = [
        'product_type_name', 'product_group_name', 'graphical_appearance_name',
        'colour_group_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
        'department_name', 'index_name', 'index_group_name', 'section_name', 'garment_group_name'
    ]
    for col in cat_cols:
        if col in articles.columns:
            articles[col] = articles[col].astype('category')
            
    return articles

def process_transactions(transactions):
    """Process Transactions: Time filtering, price scaling, and outlier removal"""
    print("--- Processing Transactions ---")
    
    # Convert to datetime
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    
    # Scale price unit
    transactions['price'] = transactions['price'] * 1000
    
    # Handle Outliers using IQR method
    if 'price' in transactions.columns:
        q1 = transactions['price'].quantile(0.25)
        q3 = transactions['price'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        initial_count = len(transactions)
        transactions = transactions[
            (transactions['price'] >= lower_bound) & (transactions['price'] <= upper_bound)
        ].copy()
        print(f"Removed {initial_count - len(transactions)} outliers.")
        
    return transactions

def run_preprocess():
    """Orchestrate the entire preprocessing workflow and save files"""
    # Load
    df_trans, df_art, df_cust = load_data()
    
    # Transform
    df_cust = process_customers(df_cust)
    df_art = process_articles(df_art)
    df_trans = process_transactions(df_trans)
    
    print("\n--- Final Dtypes Check ---")
    print(f"Transactions Shape: {df_trans.shape}")
    print(f"Customers Shape: {df_cust.shape}")
    print(f"Articles Shape: {df_art.shape}")
    
    # Save data
    print("\n--- Saving Cleaned Data ---")
    df_trans.to_parquet(PROCESSED_PATH / "transactions_cleaned.parquet", index=False)
    df_art.to_parquet(PROCESSED_PATH / "articles_cleaned.parquet", index=False)
    df_cust.to_parquet(PROCESSED_PATH / "customers_cleaned.parquet", index=False)
    
    print(f"Files saved successfully to: {PROCESSED_PATH.absolute()}")
    
    return df_trans, df_art, df_cust

if __name__ == "__main__":
    run_preprocess()
