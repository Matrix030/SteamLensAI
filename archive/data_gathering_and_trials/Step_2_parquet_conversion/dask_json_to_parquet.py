# DASK -  This program converts all the json file into schema based parquet files.
import json
import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import dask

def get_directory(app_id):
    return f'../app_details_json_2/{app_id}_data/Review_Details{app_id}.json'

def get_dirs():
    """Find all subdirectories in app_details_json folder"""
    for root, subdirs, files in os.walk('../app_details_json_2', topdown=True):
        if root == '../app_details_json_2':
            # Sort numerically by app ID
            subdirs.sort(key=lambda name: int(name.split('_')[0]))
            return subdirs
    return []

def all_app_ids(dirs):
    """Extract app IDs from directory names"""
    return [str(d.split('_')[0]) for d in dirs]

def process_app(app_id):
    """Process a single app's JSON data into parquet format
    
    This function:
    1. Loads the JSON file for an app
    2. Extracts review data
    3. Converts to a pandas DataFrame
    4. Saves as parquet with metadata
    
    Returns:
        tuple: (app_id, number_of_reviews_processed)
    """
    try:
        # Load JSON
        path = get_directory(app_id)
        with open(path, 'r') as file:
            info = json.load(file)
            
        # Process data
        data = info[app_id]['data']
        reviews = data.get('review_stats', {}).get('reviews', [])
        rows = []
        
        # Extract review data
        for review in reviews:
            author_data = review.get('author', {})
            entry = {
                'name': data.get('name'),
                'steam_appid': data.get('steam_appid'),
                'required_age': data.get('required_age'),
                'is_free': data.get('is_free'),
                'controller_support': data.get('controller_support'),
                'detailed_description': data.get('detailed_description'),
                'about_the_game': data.get('about_the_game'),
                'short_description': data.get('short_description'),
                'price_overview': data.get('price_overview', {}).get('final_formatted'),
                'metacritic_score': data.get('metacritic', {}).get('score'),
                'categories': [c.get('description') for c in data.get('categories', [])],
                'genres': [g.get('description') for g in data.get('genres', [])],
                'recommendations': data.get('recommendations', {}).get('total'),
                'achievements': data.get('achievements', {}).get('total'),
                'release_coming_soon': data.get('release_date', {}).get('coming_soon'),
                'release_date': data.get('release_date', {}).get('date'),
                'review_language': review.get('language'),
                'author_steamid': author_data.get('steamid'),
                'author_num_games_owned': author_data.get('num_games_owned'),
                'author_num_reviews': author_data.get('num_reviews'),
                'author_playtime_forever': author_data.get('playtime_forever'),
                'author_play_time_last_two_weeks': author_data.get('playtime_last_two_weeks'),
                'author_playtime_at_review': author_data.get('playtime_at_review'),
                'author_last_played': author_data.get('last_played'),
                'review': review.get('review'),
                'voted_up': review.get('voted_up'),
                'votes_up': review.get('votes_up'),
                'votes_funny': review.get('votes_funny'),
                'weighted_vote_score': float(review.get('weighted_vote_score')) 
                                     if review.get('weighted_vote_score') is not None 
                                     else None
            }
            rows.append(entry)
            
        # Create DataFrame and save
        if rows:
            # Convert to DataFrame
            df = pd.DataFrame(rows)
            df = df.where(pd.notnull(df), None)  # Replace NaN with None
            
            # Save to parquet with metadata
            output_dir = "../../parquet_output_indie"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{app_id}.parquet")
            
            # Add metadata to parquet
            table = pa.Table.from_pandas(df)
            metadata = table.schema.metadata
            metadata.update({
                b'app_id': str(app_id).encode(),
                b'total_reviews': str(len(reviews)).encode(),
                b'processing_timestamp': str(time.time()).encode(),
            })
            table = table.replace_schema_metadata(metadata)
            
            # Write with compression for smaller file size
            pq.write_table(table, output_file, compression='snappy')
            return app_id, len(reviews)
        
        return app_id, 0
        
    except FileNotFoundError:
        return None, 0
    except KeyError:
        return None, 0
    except Exception:
        return None, 0

if __name__ == "__main__":
    start_time = time.time()
    
    # Set up multiprocessing - this is what makes Dask run in parallel
    dask.config.set(scheduler='processes')
    
    # Get all app IDs to process
    dirs = get_dirs()
    app_ids = all_app_ids(dirs)
    
    # Create summary file
    with open("PARQUET_length_dask.txt", 'w') as file:
        file.write(f'Started processing at {time.ctime()}\n')
    
    # Create parallel tasks using Dask
    # ---------------------------------
    # This is the key part that makes the code parallel:
    # 1. We create a "delayed" task for each app_id 
    # 2. We add all tasks to a list
    # 3. We compute all tasks in parallel with dask.compute()
    tasks = []
    for app_id in app_ids:
        # This doesn't execute yet, just creates a task
        task = dask.delayed(process_app)(app_id)
        tasks.append(task)
        
    # Now execute all tasks in parallel using all CPU cores
    results = dask.compute(*tasks, num_workers=os.cpu_count())
    
    # Process results and write summary
    total_reviews = 0
    with open("PARQUET_length_dask.txt", 'a') as file:
        for app_id, review_count in results:
            if app_id is not None and review_count > 0:
                file.write(f'Total reviews for appId {app_id}: {review_count}\n')
                total_reviews += review_count
    
    # Write final statistics
    with open("PARQUET_length_dask.txt", 'a') as file:
        file.write(f'\nTotal Reviews Stored: {total_reviews}\n')
        file.write(f'Time Taken: {time.time() - start_time:.2f} seconds\n')
        file.write(f'Finished at: {time.ctime()}\n')
    
    print(f"Total Reviews Stored: {total_reviews}")
    print(f"Time Taken: {time.time() - start_time:.2f} seconds")