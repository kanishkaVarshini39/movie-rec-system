import pandas as pd
import numpy as np
import gc
import os

def run_data_pipeline(file_paths, output_parquet='netflix_full_processed', min_ratings=10):
    print("Starting data ingestion pipeline...")
    df_list = []
    
    for file in file_paths:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing input text file: {file}")
            
        print(f"Processing {file}...")
        raw_data = []
        with open(file, 'r') as f:
            current_movie = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_movie = int(line[:-1]) 
                else:
                    user, rating, date_str = line.split(',')
                    raw_data.append([current_movie, int(user), int(rating), date_str])

        temp_df = pd.DataFrame(raw_data, columns=['Movie_ID', 'User_ID', 'Rating', 'Date'])
        temp_df['Rating'] = temp_df['Rating'].astype(np.int8)
        temp_df['Movie_ID'] = temp_df['Movie_ID'].astype(np.int32)
        temp_df['User_ID'] = temp_df['User_ID'].astype(np.int32)
        temp_df['Date'] = pd.to_datetime(temp_df['Date'])
        
        df_list.append(temp_df)
        del raw_data
        gc.collect() 

    print("Concatenating files into a comprehensive matrix...")
    df = pd.concat(df_list, ignore_index=True)
    del df_list
    gc.collect()

    print("Applying domain-specific data filtering filters...")
    filter_movies = df['Movie_ID'].value_counts() >= min_ratings
    filter_movies = filter_movies[filter_movies].index.tolist()

    filter_users = df['User_ID'].value_counts() >= min_ratings
    filter_users = filter_users[filter_users].index.tolist()

    df = df[df['Movie_ID'].isin(filter_movies) & df['User_ID'].isin(filter_users)]
    
    print(f"Saving optimized dataframe to Parquet storage format: '{output_parquet}'...")
    df.to_parquet(output_parquet, engine='pyarrow', compression='snappy')
    print("Data processing pipeline successfully completed!")

if __name__ == "__main__":
    files = ['combined_data_1.txt', 'combined_data_2.txt', 'combined_data_3.txt', 'combined_data_4.txt']
    run_data_pipeline(files)