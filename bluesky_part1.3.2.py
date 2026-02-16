import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# 1. Load your data
def load_data():
    with open("senator_follows_map.json", "r") as f:
        follows = json.load(f)
    with open("senator_post_uris_24h.json", "r") as f:
        posts = json.load(f)
    return follows, posts

def compute_jaccard(list_a, list_b):
    """Calculates Jaccard Similarity: |A ∩ B| / |A ∪ B|"""
    set_a, set_b = set(list_a), set(list_b)
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0

def create_similarity_matrix(data_dict, sorted_senators):
    """Generates a symmetric matrix of Jaccard scores."""
    n = len(sorted_senators)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            score = compute_jaccard(data_dict[sorted_senators[i]], 
                                    data_dict[sorted_senators[j]])
            matrix[i, j] = matrix[j, i] = score
    return matrix

def run_analysis():
    follows, posts = load_data()
    # Ensure both datasets have the same senators in the same base order
    all_senators = sorted(list(follows.keys()))
    
    # Generate raw matrices
    f_matrix = create_similarity_matrix(follows, all_senators)
    p_matrix = create_similarity_matrix(posts, all_senators)

    # 2. Hierarchical Clustering (Sorting)
    # We use the Follow similarity to determine the order for BOTH heatmaps
    # Distance = 1 - Similarity
    dist_matrix = squareform(1 - f_matrix, checks=False)
    Z = linkage(dist_matrix, method='ward')
    order = leaves_list(Z)
    
    # Reorder labels and matrices
    sorted_labels = [all_senators[i] for i in order]
    f_sorted = f_matrix[order][:, order]
    p_sorted = p_matrix[order][:, order]

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Heatmap 1: Follow Similarity
    sns.heatmap(f_sorted, xticklabels=sorted_labels, yticklabels=sorted_labels, 
                ax=ax1, cmap="YlGnBu", cbar_kws={'label': 'Similarity'})
    ax1.set_title("Follow Jaccard Similarity\n(Who they follow)")

    # Heatmap 2: Post Similarity
    sns.heatmap(p_sorted, xticklabels=sorted_labels, yticklabels=sorted_labels, 
                ax=ax2, cmap="YlGnBu", cbar_kws={'label': 'Similarity'})
    ax2.set_title("Post Jaccard Similarity\n(What they actually see)")

    plt.tight_layout()
    plt.savefig("echo_chamber_heatmaps.png")
    plt.show()

if __name__ == "__main__":
    run_analysis()