#!/usr/bin/env python3


import time
from bluesky_helpers import(
    load_senators, get_author_feed, get_post_thread, 
    is_within_hours, save_json
)

## what to do:
## collect relplies to senatos posts (at least 5 female and 5 male senators)
## fetch their pots from the last 7 days
## for each post with replies, fetch the reply thread
## extract replier information (handle, display name, timestamp) and post metadata (reply count)
## save the data in jason for ecah senator


## load senators
senators = load_senators('senators_bluesky.csv')


## fetch posts for each senator in the sample
for senator in senators:
    posts = []
    cursor = None
    while True:
        result = get_author_feed(senator['handle'], limit=100, cursor=cursor)
        if not result or 'feed' not in result:
            break
        for item in result['feed']:
            created_at = item['post']['record'].get('createdAt')
            if is_within_hours(created_at, hours=168):  # 7 days = 168 hours
                posts.append(item)
            else:
                break  # posts are chronological, so we can stop
        cursor = result.get('cursor')
        if not cursor:
            break
        ## break in API calls to avoid rate limits
        time.sleep(0.1)

 #fetch reply threads for each post
    senator_data = []
    for item in posts:
        post = item['post']
        reply_count = post.get('replyCount', 0)
        if reply_count == 0:
            continue  # skip posts with no replies
        
        uri = post['uri']
        thread = get_post_thread(uri)
        time.sleep(0.1)
        
        if not thread or 'thread' not in thread:
            continue
        
        # Extract replier info from the thread
        replies = []
        for reply in thread['thread'].get('replies', []):
            reply_post = reply.get('post', {})
            replies.append({
                'handle': reply_post.get('author', {}).get('handle'),
                'displayName': reply_post.get('author', {}).get('displayName'),
                'createdAt': reply_post.get('record', {}).get('createdAt'),
                'text': reply_post.get('record', {}).get('text'),
                'likeCount': reply_post.get('likeCount', 0),
            })
        
        senator_data.append({
            'post_uri': uri,
            'post_text': post['record'].get('text'),
            'post_createdAt': post['record'].get('createdAt'),
            'replyCount': reply_count,       # total replies (important!)
            'replies_collected': len(replies), # how many we actually got
            'replies': replies,
        })

    save_json(senator_data, f"replies_{senator['handle'].replace('.', '_')}.json")

