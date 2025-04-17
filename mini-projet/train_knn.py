from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
import joblib

# Step 1: Load the user-item matrix
user_item_matrix = load_npz("/Users/zac/mini-projet/data/user_item_matrix.npz")

# Step 2: Fit kNN model on the user-item matrix
knn_model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
knn_model.fit(user_item_matrix)

joblib.dump(knn_model, '/Users/zac/mini-projet/models/knn.pkl')