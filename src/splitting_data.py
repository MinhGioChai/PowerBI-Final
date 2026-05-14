import pandas as pd
import os

def prepare_and_split_data():
    # 1. Setup Directory Paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    
    # Check if the data folder exists
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' was not found.")
        return

    # Input file paths
    transactions_path = os.path.join(data_dir, "transactions.csv")
    customers_path = os.path.join(data_dir, "customers.csv")
    articles_path = os.path.join(data_dir, "articles.csv")

    print("--- Loading data from /data folder ---")
    
    try:
        # 2. Load and Merge Data
        # Using left joins to preserve all transaction records
        transactions = pd.read_csv(transactions_path)
        customers = pd.read_csv(customers_path)
        articles = pd.read_csv(articles_path)

        # Execution of Merge: Transactions -> Customers -> Articles
        df = transactions.merge(customers, on='customer_id', how='left')
        df = df.merge(articles, on='article_id', how='left')
        
        print(f"Merged DataFrame Shape: {df.shape}")
        
        # Free up memory by deleting raw DataFrames
        del transactions, customers, articles

        # 3. Datetime Processing and Sorting
        df['t_dat'] = pd.to_datetime(df['t_dat'])
        df = df.sort_values('t_dat').reset_index(drop=True)

        min_date = df['t_dat'].min()
        max_date = df['t_dat'].max()
        
        # 4. Define Split Point (Last 7 days for testing)
        split_date = max_date - pd.Timedelta(days=7)
        
        print(f"Data Date Range: {min_date.date()} to {max_date.date()}")
        print(f"Split Threshold Date: {split_date.date()}")

        # 5. Perform Train/Test Split
        df_train = df[df['t_dat'] < split_date].copy()
        df_test = df[df['t_dat'] >= split_date].copy()

        # 6. Export to CSV within the /data folder
        train_out = os.path.join(data_dir, "train_split.csv")
        test_out = os.path.join(data_dir, "test_split.csv")

        df_train.to_csv(train_out, index=False)
        df_test.to_csv(test_out, index=False)

        print("\n--- Process Complete! ---")
        print(f"Saved Train Split: {train_out} | Shape: {df_train.shape}")
        print(f"Saved Test Split:  {test_out}  | Shape: {df_test.shape}")

    except FileNotFoundError as e:
        print(f"Error: Missing input files in the data folder. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    prepare_and_split_data()