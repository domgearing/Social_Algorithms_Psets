#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# In[2]:
df_raw = pd.read_csv('comma-survey.csv')
df_gpt = pd.read_csv('gpt_comma_survey.csv')


# In[3]:
def plot_distributions(df, columns, title_prefix="Distribution of"):
    """
    Helper function to plot bar charts for categorical columns.
    """
    for col in columns:
        if col not in df.columns:
            continue
        plt.figure(figsize=(10, 6))
        
        # Calculate percentages
        percentages = df[col].value_counts(normalize=True) * 100
        
        # Create bar plot with percentages
        ax = sns.barplot(y=percentages.index, x=percentages.values, palette="viridis")
        plt.title(f'{title_prefix}: {col.replace("_", " ").title()}')
        plt.xlabel('Percentage (%)')
        plt.ylabel('')
        
        # Add percentage labels to the end of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        
        plt.tight_layout()
        plt.show()

# %%
demo_cols = ['Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']
print("--- Plotting Demographics of raw data ---")
plot_distributions(df_raw, demo_cols, title_prefix="Demographic")
print("--- Plotting Demographics of GPT data ---")
plot_distributions(df_gpt, demo_cols, title_prefix="Demographic (GPT)")

## We do not notice a significant difference in demographics between the two datasets.