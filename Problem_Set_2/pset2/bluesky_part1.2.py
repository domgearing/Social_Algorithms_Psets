import json
import pandas as pd
from bluesky_helpers import save_json

# 1. Load the follow data collected in Part I.1
with open("senator_follows_map.json", "r") as f:
    senator_follows = json.load(f)

# Get the list of all senator handles from the keys of your JSON
all_senator_handles = set(senator_follows.keys())


def generate_recommendations():
    recommendations_output = {}

    # Precompute senator-only follow sets for efficiency
    senator_follow_sets = {
        s: set(follows).intersection(all_senator_handles)
        for s, follows in senator_follows.items()
    }

    for senator, following in senator_follows.items():
        following_set = set(following)
        senator_following_set = senator_follow_sets.get(senator, set())

        # A. Identify senators they do NOT currently follow
        # (Exclude themselves from the candidate list)
        not_followed = all_senator_handles - senator_following_set - {senator}

        # Handle edge cases: Follows everyone or follows no one
        if not not_followed:
            recommendations_output[senator] = "This senator already follows all other senators."
            continue
        if not senator_following_set:
            recommendations_output[senator] = "This senator follows no other senators."
            continue

        # B. Compute Recommendation Scores
        # Count how many other senators follow each 'not_followed' candidate
        candidate_scores = {}
        for candidate in not_followed:
            # Score = number of senators this senator follows who also follow the candidate
            score = sum(
                1
                for followed_senator in senator_following_set
                if candidate in senator_follow_sets.get(followed_senator, set())
            )
            candidate_scores[candidate] = score

        # C. Report Top 3 Recommendations
        # Sort by score (descending) and then handle (alphabetical) for stability
        top_3 = sorted(candidate_scores.items(),
                       key=lambda x: (-x[1], x[0]))[:3]

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

    return recommendations_output


def create_report_table(recs):
    """Generates the formatted table required for the assignment."""
    rows = []
    for senator, data in recs.items():
        row = {"Senator": senator}
        if isinstance(data, list):
            for i in range(3):
                val = f"{data[i]['handle']} ({data[i]['score']})" if i < len(
                    data) else "N/A"
                row[f"Rec {i+1}"] = val
        else:
            row["Rec 1"], row["Rec 2"], row["Rec 3"] = data, "N/A", "N/A"
        rows.append(row)

    df = pd.DataFrame(rows)
    print("\n### PART I.2 RECOMMENDATION TABLE")
    print(df.to_markdown(index=False))
    return df


def identify_extremes():
    """Identifies senators at the edges or center of the network."""
    num_others = len(all_senator_handles) - 1

    # In-degree: how many other senators follow them
    in_degree = {s: sum(1 for f in senator_follows.values() if s in f)
                 for s in all_senator_handles}
    # Out-degree: how many other senators they follow
    out_degree = {s: len(set(f).intersection(all_senator_handles))
                  for s, f in senator_follows.items()}

    print("\n### NETWORK ANALYSIS")
    print(
        f"Followed by ALL: {[s for s, count in in_degree.items() if count == num_others] or 'None'}")
    print(
        f"Followed by ZERO: {[s for s, count in in_degree.items() if count == 0] or 'None'}")
    print(
        f"Follows ALL: {[s for s, count in out_degree.items() if count == num_others] or 'None'}")
    print(
        f"Follows ZERO: {[s for s, count in out_degree.items() if count == 0] or 'None'}")


if __name__ == "__main__":
    # Run the full pipeline
    results = generate_recommendations()
    save_json(results, "senator_recommendations.json")
    create_report_table(results)
    identify_extremes()
