import pandas as pd
import time
from surprise import Dataset, Reader, SVD
from surprise import dump

def train_production_model(parquet_path='netflix_full_processed', model_output='svd_netflix_model.pkl'):
    print(f"Loading pre-processed data from {parquet_path}...")
    df = pd.read_parquet(parquet_path, engine='pyarrow')
    
    reader = Reader(rating_scale=(1, 5))
    print("Loading dataframe into Surprise's matrix format...")
    data = Dataset.load_from_df(df[['User_ID', 'Movie_ID', 'Rating']], reader)
    
    print("Building full global production trainset...")
    trainset = data.build_full_trainset()
    
    print("Initializing Funk SVD Model Matrix Optimization...")
    algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, verbose=True)
    
    start_time = time.time()
    algo.fit(trainset)
    print(f"Global Model Training complete! Duration: {(time.time() - start_time) / 60:.2f} mins.")
    
    print(f"Dumping trained artifact to disk path: '{model_output}'...")
    dump.dump(model_output, predictions=None, algo=algo, verbose=1)
    print("Model Training Pipeline completed successfully!")

if __name__ == "__main__":
    train_production_model()