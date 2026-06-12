import pandas as pd
from surprise import dump

class NetflixRecommender:
    def __init__(self, model_path='svd_netflix_model.pkl', metadata_csv='movie_titles.csv'):
        print("Initializing Recommendation Generation Module...")
        _, self.algo = dump.load(model_path)
        self.movies_df = pd.read_csv(metadata_csv, encoding='ISO-8859-1', 
                                     header=None, names=['Movie_ID', 'Year', 'Title'], 
                                     on_bad_lines='skip')
        self.movies_df['Year'] = self.movies_df['Year'].fillna(0).astype(int)
        self.all_movie_ids = None

    def get_top_k(self, user_id, ratings_df_path='netflix_full_processed', k=10):
        ratings_df = pd.read_parquet(ratings_df_path, engine='pyarrow')
        if self.all_movie_ids is None:
            self.all_movie_ids = set(ratings_df['Movie_ID'].unique())
            
        user_history = ratings_df[ratings_df['User_ID'] == user_id]
        watched_movie_ids = set(user_history['Movie_ID'])
        unseen_movie_ids = list(self.all_movie_ids - watched_movie_ids)
        
        if not unseen_movie_ids:
            return pd.DataFrame(columns=['Movie_ID', 'Title', 'Year', 'Predicted_Rating'])
            
        predictions = []
        for movie_id in unseen_movie_ids:
            pred = self.algo.predict(uid=user_id, iid=movie_id)
            predictions.append((movie_id, pred.est))
            
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_preds = predictions[:k]
        
        recs = []
        for movie_id, estimated_rating in top_k_preds:
            movie_info = self.movies_df[self.movies_df['Movie_ID'] == movie_id].iloc[0]
            recs.append({
                'Movie_ID': movie_id,
                'Title': movie_info['Title'],
                'Year': movie_info['Year'],
                'Predicted_Rating': round(estimated_rating, 3)
            })
        return pd.DataFrame(recs)

if __name__ == "__main__":
    # Smoke test execution
    engine = NetflixRecommender()
    sample_recs = engine.get_top_k(user_id=30878, k=5)
    print("\nSample Top-5 Recommendations for User 30878:")
    print(sample_recs)