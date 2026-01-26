
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import numpy as np

survey_path = 'Problem_Set_1/comma-survey.csv'
gpt_path = 'Problem_Set_1/gpt_comma_survey.csv'
survey_df = pd.read_csv(survey_path)
gpt_df = pd.read_csv(gpt_path)
survey_df = survey_df.rename(columns={
    'RespondentID': 'id',
    'In your opinion, which sentence is more gramatically correct?': 'comma_preference',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?': 'heard_of_comma',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?': 'care_oxford_comma',
    'How would you write the following sentence?': 'sentence_preference',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?': 'data_singular_plural_consideration',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?': 'care_data_debate',
    'In your opinion, how important or unimportant is proper use of grammar?': 'grammar_importance'
})
gpt_df = gpt_df.rename(columns={
    'RespondentID': 'id',
    'In your opinion, which sentence is more gramatically correct?': 'comma_preference',
    'Prior to reading about it above, had you heard of the serial (or Oxford) comma?': 'heard_of_comma',
    'How much, if at all, do you care about the use (or lack thereof) of the serial (or Oxford) comma in grammar?': 'care_oxford_comma',
    'How would you write the following sentence?': 'sentence_preference',
    'When faced with using the word "data", have you ever spent time considering if the word was a singular or plural noun?': 'data_singular_plural_consideration',
    'How much, if at all, do you care about the debate over the use of the word "data" as a singluar or plural noun?': 'care_data_debate',
    'In your opinion, how important or unimportant is proper use of grammar?': 'grammar_importance'
})

#setup logging to capture all output
log_filename = f"logs/census_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

import sys
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

#load census data
census_path = 'Problem_Set_1/post_strat_long_full_cartesian_FIXED_v2.csv'
census_df = pd.read_csv(census_path)

#map census columns to survey demographic columns
census_df = census_df.rename(columns={
    'sex': 'Gender',
    'age_group': 'Age',
    'income_bin': 'Household Income',
    'education_5': 'Education',
    'census_region': 'Location (Census Region)'
})

def plot_stacked_demographic_comparison(census_df, survey_df, gpt_df, demo_cols, filename_prefix="stacked_compare_"):
    import matplotlib.pyplot as plt
    import numpy as np
    # For each demographic variable
    for idx, col in enumerate(demo_cols):
        # Census percentages
        census_pop_sum = census_df['pop_count'].sum()
        census_pct = census_df.groupby(col)['pop_count'].sum() / census_pop_sum * 100
        # Human survey percentages
        survey_pct = survey_df[col].value_counts(normalize=True) * 100
        # GPT survey percentages
        gpt_pct = gpt_df[col].value_counts(normalize=True) * 100
        # Align all categories
        all_cats = sorted(set(census_pct.index) | set(survey_pct.index) | set(gpt_pct.index))
        census_vals = [census_pct.get(cat, 0) for cat in all_cats]
        survey_vals = [survey_pct.get(cat, 0) for cat in all_cats]
        gpt_vals = [gpt_pct.get(cat, 0) for cat in all_cats]
        # Stack for 100% bar chart
        data = np.array([census_vals, survey_vals, gpt_vals])
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(3)
        colors = plt.cm.viridis(np.linspace(0, 1, len(all_cats)))
        for i, cat in enumerate(all_cats):
            vals = data[:, i]
            ax.bar(["Census", "Human Survey", "GPT Survey"], vals, bottom=bottom, color=colors[i], label=cat)
            bottom += vals
        ax.set_ylabel("Percentage of Population")
        ax.set_title(f"100% Stacked Comparison: {col}")
        ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        sanitized_col_name = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f'Problem_Set_1/viz_census/{filename_prefix}{idx+1:02d}_{sanitized_col_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def plot_distributions(df, columns, title_prefix="Distribution of", filename_prefix=""):
    for idx, col in enumerate(columns):
        if col not in df.columns:
            continue
        # Calculate percentage of total population for each category
        pop_sum = df['pop_count'].sum()
        pct = df.groupby(col)['pop_count'].sum() / pop_sum * 100
        pct = pct.sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        order = pct.index.tolist()
        colors = plt.cm.viridis(np.linspace(0, 1, len(order)))
        plt.barh(order, pct.values, color=colors)
        plt.title(f'{title_prefix} (%): {col.replace("_", " ").title()}')
        plt.xlabel('Percentage of Total Population')
        plt.ylabel('')
        for i, v in enumerate(pct.values):
            plt.text(v + 0.5, i, f'{v:.2f}%', va='center')
        plt.tight_layout()
        sanitized_col_name = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f'Problem_Set_1/viz_census/{filename_prefix}{idx+1:02d}_{sanitized_col_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

demo_cols = ['Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']
print("--- Plotting Census Demographics ---")
plot_distributions(census_df, demo_cols, title_prefix="Census Demographic", filename_prefix="census_demo_")

print("--- Plotting 100% Stacked Demographic Comparisons ---")
plot_stacked_demographic_comparison(census_df, survey_df, gpt_df, demo_cols)

print("Done. Visualizations saved in viz_census/")
