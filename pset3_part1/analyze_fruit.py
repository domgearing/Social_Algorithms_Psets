import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('fruit_results.csv')

def calculate_entropy(series):
    probs = series.value_counts() / len(series)
    return -np.sum(probs * np.log2(probs))

fig, axes = plt.subplots(len(df['temperature'].unique()), 1, figsize=(10, 12))
plt.subplots_adjust(hspace=0.5)

for i, t in enumerate(sorted(df['temperature'].unique())):
    subset = df[df['temperature'] == t]
    counts = subset['fruit'].value_counts().head(10) # Top 10 fruits
    
    counts.plot(kind='bar', ax=axes[i], color='salmon')
    axes[i].set_title(f"Temperature {t} (Entropy: {calculate_entropy(subset['fruit']):.4f})")
    axes[i].set_ylabel("Frequency")

plt.savefig("fruit_histograms.png")
print("Analysis complete. Plot saved as fruit_histograms.png")

# Print metrics for your report
for t in sorted(df['temperature'].unique()):
    subset = df[df['temperature'] == t]
    print(f"T={t} | Unique Fruits: {subset['fruit'].nunique()} | Entropy: {calculate_entropy(subset['fruit']):.4f}")