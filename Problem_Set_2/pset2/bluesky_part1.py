import pandas as pd
import time
import json
from datetime import datetime, timezone, timedelta
from bluesky_helpers import get_follows, get_author_feed, is_within_hours, save_json

# 1. Load the Senator Data
# Replace 'senators.csv' with your actual filename
df = pd.read_csv('/Users/tadcarney/Desktop/s&ds_3350/pset2/senators_bluesky.csv')

def collect_feeds():
    all_follow_data = {}
    
    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        senator_handle = row['handle']
        senator_name = row['name']
        
        print(f"[{index+1}/{len(df)}] Processing: {senator_name} ({senator_handle})")
        
        # --- I.1.a: Retrieve Follows (with Pagination) ---
        followed_handles = []
        cursor = None
        while True:
            result = get_follows(senator_handle, limit=100, cursor=cursor)
            if not result: break
            
            followed_handles.extend([f['handle'] for f in result.get('follows', [])])
            cursor = result.get('cursor')
            if not cursor: break
            time.sleep(0.1)
        
        # Store follows for Part I.3 Jaccard Similarity
        all_follow_data[senator_handle] = followed_handles
        
        # --- I.1.b: Fetch 24-hour Feed ---
        senator_combined_feed = []
        for account in followed_handles:
            feed_data = get_author_feed(account)
            if feed_data and 'feed' in feed_data:
                for item in feed_data['feed']:
                    post = item.get('post', {})
                    created_at = post.get('record', {}).get('createdAt')
                    
                    if created_at and is_within_hours(created_at, hours=24):
                        # Append post + senator metadata for easier analysis later
                        senator_combined_feed.append({
                            'senator_handle': senator_handle,
                            'senator_party': row['party'], # Accessing columns from the CSV
                            'author': account,
                            'text': post.get('record', {}).get('text'),
                            'createdAt': created_at,
                            'uri': post.get('uri')
                        })
            time.sleep(0.1)

        # Sort and Save individual feed
        senator_combined_feed.sort(key=lambda x: x['createdAt'], reverse=True)
        save_json(senator_combined_feed, f"feed_{senator_handle.replace('.', '_')}.json")
        
    # Save global follow mapping
    save_json(all_follow_data, "senator_follows_map.json")

if __name__ == "__main__":
    collect_feeds()