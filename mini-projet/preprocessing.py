import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import os

train = pd.read_csv("/Users/zac/mini-projet/data/train.csv")
test = pd.read_csv("/Users/zac/mini-projet/data/test.csv")

# Step 1: Load the transactions data
transactions = pd.read_csv("/Users/zac/mini-projet/data/transactions_train_reduced.csv", parse_dates=["t_dat"])

# Step 2: Sort transactions by user and timestamp
transactions = transactions.sort_values(["customer_id", "t_dat"])

# Step 3: Encode user and item IDs based on training data
user_encoder = transactions["customer_id"].astype("category")
item_encoder = transactions["article_id"].astype("category")

train["user_index"] = user_encoder.cat.codes
train["item_index"] = item_encoder.cat.codes

# Map test set using same categories
test["user_index"] = test["customer_id"].astype("category").cat.set_categories(user_encoder.cat.categories).cat.codes
test["item_index"] = test["article_id"].astype("category").cat.set_categories(item_encoder.cat.categories).cat.codes

# Drop unknown users/items (if any were not seen in train)
test = test[(test["user_index"] != -1) & (test["item_index"] != -1)]

# Step 4: Build the user-item sparse matrix
user_item_matrix = coo_matrix(
    (np.ones(len(train)), (train["user_index"], train["item_index"]))
)

# Step 5: Save the user-item matrix
os.makedirs("data_hm/splits", exist_ok=True)
from scipy.sparse import save_npz
save_npz("/Users/zac/mini-projet/data/user_item_matrix.npz", user_item_matrix)

# Optionally save encoders for future use (e.g., decoding the indices back to original IDs)
user_id_map = dict(enumerate(user_encoder.cat.categories))
item_id_map = dict(enumerate(item_encoder.cat.categories))

pd.to_pickle(user_id_map, "/Users/zac/mini-projet/data/user_id_map.pkl")
pd.to_pickle(item_id_map, "/Users/zac/mini-projet/data/item_id_map.pkl")
