import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('day_of_week_results.csv')
days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
temps = [0.1, 0.5, 1.0, 1.5, 2.0]

for i, t in enumerate(temps):
    subset = df[df['temperature'] == t]['answer']
    counts = subset.value_counts().reindex(days, fill_value=0)
    
    axes[i].bar(counts.index, counts.values, color='skyblue', edgecolor='black')
    axes[i].set_title(f"Temp: {t}")
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('histograms.png')
print("Plot created: histograms.png")

def calculate_entropy(series):
    # Get the counts of each day
    counts = series.value_counts()
    # Convert to probabilities (frequencies)
    probs = counts / len(series)
    # Shannon Entropy formula: -sum(p * log2(p))
    return -np.sum(probs * np.log2(probs))

for t in df['temperature'].unique():
    subset = df[df['temperature'] == t]
    entropy = calculate_entropy(subset['answer'])
    unique_days = subset['answer'].nunique()
    print(f"Temp {t}: Entropy = {entropy:.4f} | Unique Days = {unique_days}")