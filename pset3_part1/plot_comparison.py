import pandas as pd
import matplotlib.pyplot as plt

# Load the comparison data
df = pd.read_csv('prompt_comparison.csv')
prompt_types = df['prompt_type'].unique()

# Create subplots for each prompt style
fig, axes = plt.subplots(len(prompt_types), 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.6)

# Handle cases where there might only be one prompt type
if len(prompt_types) == 1:
    axes = [axes]

for i, p_type in enumerate(prompt_types):
    subset = df[df['prompt_type'] == p_type]
    # Get top 10 fruits to keep the chart readable
    counts = subset['fruit'].value_counts().head(10)
    
    counts.plot(kind='bar', ax=axes[i], color='skyblue')
    axes[i].set_title(f"Prompt Strategy: {p_type}")
    axes[i].set_ylabel("Frequency")
    axes[i].set_xlabel("Fruit Name")

# Save the final comparison image
plt.savefig('comparison_histograms.png')
print("Comparison histograms saved as comparison_histograms.png")