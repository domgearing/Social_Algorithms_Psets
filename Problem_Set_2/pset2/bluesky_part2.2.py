#!/usr/bin/env python3
from bluesky_helpers import(
    load_name_data, infer_gender, load_json, load_senators
)
# for reading in json files
import os 

# 1. Load SSA name data
name_data = load_name_data()

# 2. Load senator info (so we know each senator's gender)
senators = load_senators('senators_bluesky.csv')
senator_gender = {s['handle']: s['gender'] for s in senators}

# 3. Load all reply JSON files and run gender inference
results = {}  # senator_handle -> list of inferred genders

reply_files = [f for f in os.listdir('.') if f.startswith('replies_') and f.endswith('.json')]

total_repliers = 0
classified_f = 0
classified_m = 0
classified_u = 0

for filename in reply_files:
    data = load_json(filename)
    
    for post in data:
        for reply in post['replies']:
            display_name = reply.get('displayName', '')
            gender = infer_gender(display_name, name_data)
            reply['inferred_gender'] = gender  # store it for later use
            
            total_repliers += 1
            if gender == 'F':
                classified_f += 1
            elif gender == 'M':
                classified_m += 1
            else:
                classified_u += 1

# 4. Report classification results
classified = classified_f + classified_m
print(f"Total repliers: {total_repliers}")
print(f"Classified:     {classified} ({classified/total_repliers*100:.1f}%)")
print(f"  Female:       {classified_f} ({classified_f/total_repliers*100:.1f}%)")
print(f"  Male:         {classified_m} ({classified_m/total_repliers*100:.1f}%)")
print(f"  Unknown:      {classified_u} ({classified_u/total_repliers*100:.1f}%)")
