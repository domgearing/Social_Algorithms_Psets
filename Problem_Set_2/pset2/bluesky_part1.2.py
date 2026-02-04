import json
from collections import Counter

# 1. Load the follow data collected in Part I.1
with open("senator_follows_map.json", "r") as f:
    senator_follows = json.load(f)

# Get the list of all senator handles from the keys of your JSON
all_senator_handles = set(senator_follows.keys())

def generate_recommendations():
    recommendations_output = {}

    for senator, following in senator_follows.items():
        following_set = set(following)
        
        # A. Identify senators they do NOT currently follow
        # (Exclude themselves from the candidate list)
        not_followed = all_senator_handles - following_set - {senator}
        
        # Handle edge cases: Follows everyone or follows no one
        if not not_followed:
            recommendations_output[senator] = "This senator already follows all other senators."
            continue
        if not following:
            recommendations_output[senator] = "This senator follows no other senators."
            continue

        # B. Compute Recommendation Scores
        # Count how many other senators follow each 'not_followed' candidate
        candidate_scores = {}
        for candidate in not_followed:
            # Score = number of other senators (excluding the current one) who follow the candidate
            score = sum(1 for other_senator, others_following in senator_follows.items() 
                        if other_senator != senator and candidate in others_following)
            candidate_scores[candidate] = score

        # C. Report Top 3 Recommendations
        # Sort by score (descending) and then handle (alphabetical) for stability
        top_3 = sorted(candidate_scores.items(), key=lambda x: (-x[1], x[0]))[:3]
        
        recommendations_output[senator] = [
            {"handle": rec[0], "score": rec[1]} for rec in top_3
        ]

    # Save the results
    with open("senator_recommendations.json", "w") as f:
        json.dump(recommendations_output, f, indent=4)
        
    # Print a sample for the first senator to verify
    first_senator = list(recommendations_output.keys())[0]
    print(f"Top recommendations for {first_senator}:")
    print(json.dumps(recommendations_output[first_senator], indent=2))

if __name__ == "__main__":
    generate_recommendations()