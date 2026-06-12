import pandas as pd
import time
from collections import defaultdict
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

def calculate_map_at_k(predictions, k=10, threshold=4.0):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    average_precisions = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        if n_rel == 0: 
            continue 
        top_k = user_ratings[:k]
        hits = 0
        sum_precisions = 0
        for i, (est, true_r) in enumerate(top_k):
            if true_r >= threshold:
                hits += 1
                sum_precisions += hits / (i + 1.0)
        average_precisions.append(sum_precisions / min(n_rel, k))
        
    return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0

def run_evaluation_comparison():
    print("Loading data subset for comparative evaluation runs...")
    raw_data = []
    with open('combined_data_1.txt', 'r') as f:
        current_movie = None
        for line in f:
            line = line.strip()
            if line.endswith(':'):
                current_movie = int(line[:-1])
            else:
                user, rating, _ = line.split(',')
                raw_data.append([current_movie, int(user), int(rating)])

    df_comp = pd.DataFrame(raw_data, columns=['Movie_ID', 'User_ID', 'Rating'])
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_comp[['User_ID', 'Movie_ID', 'Rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # 1. Funk SVD Evaluation
    print("\nEvaluating Model A: Funk SVD...")
    model_a = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    start = time.time()
    model_a.fit(trainset)
    svd_train_time = time.time() - start
    
    start = time.time()
    preds_a = model_a.test(testset)
    svd_inf_time = time.time() - start
    svd_rmse = accuracy.rmse(preds_a, verbose=False)
    svd_map = calculate_map_at_k(preds_a)

    # 2. KNN Evaluation
    print("Evaluating Model B: Item-KNN Baseline...")
    sim_options = {'name': 'pearson_baseline', 'user_based': False, 'min_support': 5}
    model_b = KNNBasic(sim_options=sim_options, verbose=False)
    start = time.time()
    model_b.fit(trainset)
    knn_train_time = time.time() - start
    
    start = time.time()
    preds_b = model_b.test(testset)
    knn_inf_time = time.time() - start
    knn_rmse = accuracy.rmse(preds_b, verbose=False)
    knn_map = calculate_map_at_k(preds_b)

    # Summary table creation
    comparison_data = {
        "Metric / Feature": ["RMSE ↓", "MAP@10 ↑", "Training Latency", "Inference Latency"],
        "Funk SVD (Matrix Factorization)": [f"{svd_rmse:.4f}", f"{svd_map:.4f}", f"{svd_train_time:.2f}s", f"{svd_inf_time:.2f}s"],
        "Item-Based KNN": [f"{knn_rmse:.4f}", f"{knn_map:.4f}", f"{knn_train_time:.2f}s", f"{knn_inf_time:.2f}s"]
    }
    print("\n=== SYSTEM PERFORMANCE MATRIX ===")
    print(pd.DataFrame(comparison_data).to_string(index=False))

if __name__ == "__main__":
    run_evaluation_comparison()