#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime

# Create comparison_viz and logs folders
os.makedirs('comparison_viz', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Setup logging to capture all output
log_filename = f"logs/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, 'w')

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
df_raw = pd.read_csv('comma-survey.csv')
df_gpt = pd.read_csv('gpt_comma_survey.csv')

# Rename columns for both datasets
column_mapping = {
    'RespondentID': 'id',
    'In your opinion, which sentence is more gramatically correct?': 'comma_preference',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?': 'heard_of_comma',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?': 'care_oxford_comma',
    'How would you write the following sentence?': 'sentence_preference',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?': 'data_singular_plural_consideration',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?': 'care_data_debate',
    'In your opinion, how important or unimportant is proper use of grammar?': 'grammar_importance'
}

df_raw.rename(columns=column_mapping, inplace=True)
df_gpt.rename(columns=column_mapping, inplace=True)



question_cols = [
    'comma_preference', 
    'heard_of_comma', 
    'care_oxford_comma', 
    'sentence_preference',
    'data_singular_plural_consideration',
    'care_data_debate',
    'grammar_importance'
]


# In[2]:
# Define valid responses for each question -- GPT gave some slightly off answers, 
# so we used AI to generate this code that matches based on key distinguishing content.

valid_responses = {
    'comma_preference': [
        "It's important for a person to be honest, kind and loyal.",
        "It's important for a person to be honest, kind, and loyal."
    ],
    'heard_of_comma': ["Yes", "No"],
    'care_oxford_comma': ["A lot", "Some", "Not much", "Not at all"],
    'sentence_preference': [
        "Some experts say it's important to drink milk, but the data are inconclusive.",
        "Some experts say it's important to drink milk, but the data is inconclusive."
    ],
    'data_singular_plural_consideration': ["Yes", "No"],
    'care_data_debate': ["A lot", "Some", "Not much", "Not at all"],
    'grammar_importance': ["Very important", "Somewhat important", "Somewhat unimportant", "Very unimportant"]
}

def match_to_valid_fuzzy(response, valid_options, col):
    """
    Match based on key distinguishing content.
    """
    if pd.isna(response):
        return response
    
    cleaned = response.lower()
    
    # For comma_preference - check if "kind, and" (with comma) or "kind and" (without)
    if col == 'comma_preference':
        if 'kind, and' in cleaned:
            return "It's important for a person to be honest, kind, and loyal."
        elif 'kind and' in cleaned:
            return "It's important for a person to be honest, kind and loyal."
    
    # For sentence_preference - check "data are" vs "data is"
    if col == 'sentence_preference':
        if 'data are' in cleaned:
            return "Some experts say it's important to drink milk, but the data are inconclusive."
        elif 'data is' in cleaned:
            return "Some experts say it's important to drink milk, but the data is inconclusive."
    
    # For simple yes/no or short answers - strip quotes and periods, then match
    cleaned_simple = cleaned.strip().strip('"').strip("'").strip('.').strip()
    for valid in valid_options:
        if valid.lower() == cleaned_simple:
            return valid
    
    print(f"No match found for: {repr(response)}")
    return response

# Apply the mapping to each column
for col in question_cols:
    df_gpt[col] = df_gpt[col].apply(
        lambda x: match_to_valid_fuzzy(x, valid_responses[col], col)
    )



# In[2]:
def plot_comparison(df_human, df_gpt, columns):
    """
    Plot side-by-side comparison of response distributions for human vs GPT data.
    """
    for idx, col in enumerate(columns):
        if col not in df_human.columns or col not in df_gpt.columns:
            continue
        
        # Calculate percentages for both datasets
        human_pct = df_human[col].value_counts(normalize=True) * 100
        gpt_pct = df_gpt[col].value_counts(normalize=True) * 100
        
        # Get all unique responses from both datasets
        all_responses = list(set(human_pct.index) | set(gpt_pct.index))
        
        # Create a combined dataframe for plotting
        comparison_df = pd.DataFrame({
            'Response': all_responses * 2,
            'Percentage': [human_pct.get(r, 0) for r in all_responses] + [gpt_pct.get(r, 0) for r in all_responses],
            'Source': ['Human'] * len(all_responses) + ['GPT'] * len(all_responses)
        })
        
        # Plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=comparison_df, y='Response', x='Percentage', hue='Source', palette=['steelblue', 'coral'])
        plt.title(f'Human vs GPT: {col.replace("_", " ").title()}')
        plt.xlabel('Percentage (%)')
        plt.ylabel('')
        
        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%')
        
        plt.legend(title='Source')
        plt.tight_layout()
        
        # Save as PNG
        sanitized_col_name = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f'comparison_viz/{idx+1:02d}_{sanitized_col_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print the raw numbers for the report
        print(f"\n--- {col} ---")
        print("Human responses:")
        print(human_pct.round(1))
        print("\nGPT responses:")
        print(gpt_pct.round(1))
        print()

# Run the comparison
plot_comparison(df_raw, df_gpt, question_cols)
# %%


# %%
