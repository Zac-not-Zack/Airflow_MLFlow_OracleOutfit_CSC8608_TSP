import pandas as pd
from scipy.sparse import load_npz
import joblib
import mlflow
import mlflow.sklearn

# Load the preprocessed data
user_item_matrix = load_npz("/Users/zac/mini-projet/data/user_item_matrix.npz")  # Ensure this path is correct
test_df = pd.read_csv("/Users/zac/mini-projet/data/test.csv")
transactions = pd.read_csv("/Users/zac/mini-projet/data/transactions_train_reduced.csv", parse_dates=["t_dat"])

# Encoding user and item indices (ensure consistency with training data)
user_encoder = transactions["customer_id"].astype("category")
item_encoder = transactions["article_id"].astype("category")

# Create mappings for user and item indices in the test data
test_df["user_index"] = test_df["customer_id"].astype("category").cat.set_categories(user_encoder.cat.categories).cat.codes
test_df["item_index"] = test_df["article_id"].astype("category").cat.set_categories(item_encoder.cat.categories).cat.codes

# Load the trained kNN model
knn_model = joblib.load("/Users/zac/mini-projet/models/knn.pkl")

# Define the evaluation function
def evaluate_knn_model(knn_model, user_item_matrix, test_df, k=10):
    hit_count = 0
    total_users = 0
    precision_sum = 0
    recall_sum = 0

    for _, row in test_df.iterrows():
        user_idx = row["user_index"]
        true_item_idx = row["item_index"]

        # Get the sparse user vector (row) from user_item_matrix
        user_vector = user_item_matrix.getrow(user_idx)

        # Find k similar items (for the user)
        distances, indices = knn_model.kneighbors(user_vector, n_neighbors=k)

        # Flatten to array (list of recommended items)
        recommended_items = indices.flatten()

        # Evaluate whether the true item is among the recommended items
        hit = true_item_idx in recommended_items
        hit_count += hit
        precision_sum += hit / k
        recall_sum += hit  # Only one relevant item per user in test set
        total_users += 1

    precision_at_k = precision_sum / total_users
    recall_at_k = recall_sum / total_users

    # Log Precision and Recall to MLflow (use valid metric names)
    mlflow.log_metric('Precision_at_k', precision_at_k)
    mlflow.log_metric('Recall_at_k', recall_at_k)

    print(f"✅ Evaluation for k = {k}")
    print(f"Precision@{k}: {precision_at_k:.4f}")
    print(f"Recall@{k}:    {recall_at_k:.4f}")
    return precision_at_k, recall_at_k

# Set up MLflow tracking
mlflow.set_tracking_uri('http://127.0.0.1:5000')  # Adjust if needed for your server
mlflow.start_run()

# Run the evaluation and get Precision/Recall
precision_at_k, recall_at_k = evaluate_knn_model(knn_model, user_item_matrix, test_df=test_df, k=10)

# Log model and metrics in MLflow
mlflow.log_param('Model', knn_model.__class__.__name__)  # Log the model class
mlflow.sklearn.log_model(knn_model, "knn_model")  # Log the trained kNN model

# Optional: Log the best score (precision_at_k or recall_at_k)
best_score = precision_at_k  # Or select based on your preference
mlflow.log_metric('Best_Score', best_score)

mlflow.end_run()
