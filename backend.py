from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="Netflix Recommendation Engine API")

# --- GLOBAL DATA AND MODEL LOADING ON STARTUP ---
MOVIES_CSV = "movie_titles.csv"
INTERACTIONS_PARQUET = "netflix_full_processed.parquet"
MODEL_WEIGHTS_PKL = "svd_netflix_model.pkl"

# 1. Load DataFrames
movies_df = pd.read_csv(MOVIES_CSV, encoding='ISO-8859-1', header=None, 
                        names=['Movie_ID', 'Year', 'Title'], on_bad_lines='skip')
movies_df['Movie_ID'] = movies_df['Movie_ID'].astype(int)

if os.path.exists(INTERACTIONS_PARQUET):
    interactions_df = pd.read_parquet(INTERACTIONS_PARQUET)
    interactions_df.columns = ['Movie_ID', 'User_ID', 'Rating', 'Date']
    interactions_df['Date'] = interactions_df['Date'].astype(str)
else:
    raise FileNotFoundError(f"Missing required historical data file: {INTERACTIONS_PARQUET}")

# 2. Compute Global Popularity Fallback for Cold Start Users
popularity_summary = interactions_df.groupby('Movie_ID').agg(
    avg_rating=('Rating', 'mean'),
    vote_count=('Rating', 'count')
).reset_index()
top_popular_movies = popularity_summary[popularity_summary['vote_count'] >= 100]\
    .sort_values(by='avg_rating', ascending=False).head(10)['Movie_ID'].tolist()

# 3. Load the Surprise Algorithm Object
print("Loading Surprise model...")
with open(MODEL_WEIGHTS_PKL, 'rb') as f:
    model_data = pickle.load(f)

algo = model_data['algo']
trainset = algo.trainset

# --- REQUEST SCHEMAS ---
class RatingSubmission(BaseModel):
    user_id: int
    movie_id: int
    rating: int

# --- API ENDPOINTS ---

@app.get("/api/v1/recommendations/{user_id}")
def get_recommendations(user_id: int):
    global interactions_df
    
    # Check if user exists
    if user_id not in interactions_df['User_ID'].values:
        # COLD START TRIGGER: Return overall top popular films
        rec_ids = top_popular_movies
    else:
        # Get all unique movies
        all_movies = movies_df['Movie_ID'].unique()
        
        # Filter out movies the user has already watched
        watched_movies = set(interactions_df[interactions_df['User_ID'] == user_id]['Movie_ID'])
        unwatched_movies = [m for m in all_movies if m not in watched_movies]
        
        # Use Surprise to predict ratings for all unwatched movies
        predictions = [algo.predict(user_id, m_id) for m_id in unwatched_movies]
        
        # Sort by estimated rating (est) descending
        predictions.sort(key=lambda x: x.est, reverse=True)
        rec_ids = [pred.iid for pred in predictions[:10]]
        
    result = movies_df[movies_df['Movie_ID'].isin(rec_ids)].to_dict(orient='records')
    return result


@app.get("/api/v1/movies/{movie_id}/similar")
def get_similar_movies(movie_id: int):
    # Check if movie was in the training set
    try:
        inner_id = trainset.to_inner_iid(movie_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Movie not found in training set.")
        
    # Extract the item latent factors matrix (Q) from Surprise
    item_factors = algo.qi 
    target_vector = item_factors[inner_id].reshape(1, -1)
    
    # Compute similarity against all items
    similarities = cosine_similarity(target_vector, item_factors).flatten()
    sorted_inner_indices = np.argsort(similarities)[::-1]
    
    rec_ids = []
    for idx in sorted_inner_indices:
        # Convert Surprise internal ID back to raw dataset Movie_ID
        raw_m_id = trainset.to_raw_iid(idx) 
        if raw_m_id == movie_id:
            continue # Skip self
        rec_ids.append(raw_m_id)
        if len(rec_ids) == 10:
            break
            
    result = movies_df[movies_df['Movie_ID'].isin(rec_ids)].to_dict(orient='records')
    return result


@app.post("/api/v1/ratings")
def post_new_rating(payload: RatingSubmission):
    global interactions_df
    
    user_id = payload.user_id
    movie_id = payload.movie_id
    new_rating = payload.rating
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    match_condition = (interactions_df['User_ID'] == user_id) & (interactions_df['Movie_ID'] == movie_id)
    
    if match_condition.any():
        interactions_df.loc[match_condition, 'Rating'] = new_rating
        interactions_df.loc[match_condition, 'Date'] = current_date
        status_msg = "Updated existing rating successfully."
    else:
        new_row = pd.DataFrame([[movie_id, user_id, new_rating, current_date]], 
                               columns=['Movie_ID', 'User_ID', 'Rating', 'Date'])
        interactions_df = pd.concat([interactions_df, new_row], ignore_index=True)
        status_msg = "Created new rating row record successfully."
        
    # Save back to parquet
    interactions_df.to_parquet(INTERACTIONS_PARQUET, index=False)
    
    return {"status": "success", "message": status_msg}