import pandas as pd
import os

# Step 1: Load your transactions data
transactions = pd.read_csv("/Users/zac/mini-projet/data/transactions_train_reduced.csv", parse_dates=["t_dat"])

# Step 2: Sort transactions by user and timestamp
transactions = transactions.sort_values(["customer_id", "t_dat"])

# Step 3: Function to split train/test per user
def split_train_test_per_user(df):
    train_rows = []
    test_rows = []
    
    for cust_id, group in df.groupby("customer_id"):
        if len(group) < 2:
            train_rows.append(group)
        else:
            train_rows.append(group.iloc[:-1])  # all but last
            test_rows.append(group.iloc[-1:])   # last only
    
    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df = pd.concat(test_rows).reset_index(drop=True)
    return train_df, test_df

# Step 4: Split the data
train_df, test_df = split_train_test_per_user(transactions)

# Step 5: Save to CSV
os.makedirs("/Users/zac/mini-projet/data", exist_ok=True)  # ensure folder exists
train_df.to_csv("/Users/zac/mini-projet/data/train.csv", index=False)
test_df.to_csv("/Users/zac/mini-projet/data/test.csv", index=False)

print("✅ Split completed")

