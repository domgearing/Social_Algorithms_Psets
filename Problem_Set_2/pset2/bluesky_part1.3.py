import time
import json
from bluesky_helpers import get_author_feed, is_within_hours, save_json

# Load your existing follow map
with open("senator_follows_map.json", "r") as f:
    senator_follows = json.load(f)


def collect_senator_feed_uris():
    """Aggregates all post URIs seen by each senator in the last 24 hours."""
    senator_feeds_content = {}

    # We cache author feeds so we don't fetch the same person 50 times
    # (Multiple senators likely follow the same popular accounts)
    author_cache = {}

    for senator, followed_list in senator_follows.items():
        print(f"Processing feed for: {senator}...")
        all_uris = set()

        for handle in followed_list:
            if handle not in author_cache:
                try:
                    cursor = None
                    uris = []
                    stop = False

                    while True:
                        feed = get_author_feed(
                            handle, limit=100, cursor=cursor)
                        if not feed or 'feed' not in feed:
                            break

                        for item in feed.get('feed', []):
                            created_at = item['post']['record']['createdAt']
                            if is_within_hours(created_at, 24):
                                uris.append(item['post']['uri'])
                            else:
                                stop = True
                                break

                        if stop:
                            break

                        cursor = feed.get('cursor')
                        if not cursor:
                            break

                        time.sleep(0.15)

                    author_cache[handle] = uris
                    time.sleep(0.15)
                except Exception:
                    author_cache[handle] = []

            all_uris.update(author_cache[handle])

        senator_feeds_content[senator] = list(all_uris)

    save_json(senator_feeds_content, "senator_post_uris_24h.json")
    return senator_feeds_content


if __name__ == "__main__":
    print("Starting data collection...")
    collect_senator_feed_uris()
    print("Data collection complete!")
