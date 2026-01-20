#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
df = pd.read_csv('comma-survey.csv')


# In[3]:


df.head()
df.columns


# In[10]:


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
df.rename(columns=column_mapping, inplace=True)


# In[5]:


def analyze_missingness(df):
    """
    Analyzes and visualizes missing data.
    """
    print("--- Missingness Analysis ---")
    missing_counts = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({'Missing Count': missing_counts, 'Missing %': missing_pct})
    print(missing_df[missing_df['Missing Count'] > 0])
    print("\n")

    # Visualizing Missingness Pattern
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Missing Data Heatmap (Yellow = Missing)')
    plt.tight_layout()
    plt.show()


# In[6]:


def plot_distributions(df, columns, title_prefix="Distribution of"):
    """
    Helper function to plot bar charts for categorical columns.
    """
    for col in columns:
        if col not in df.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Calculate value counts and percentages
        counts = df[col].value_counts(normalize=True).sort_index()
        
        # Create bar plot
        ax = sns.countplot(y=col, data=df, order=df[col].value_counts().index, palette="viridis")
        
        plt.title(f'{title_prefix}: {col.replace("_", " ").title()}')
        plt.xlabel('Count')
        plt.ylabel('')
        
        # Add labels to the end of bars
        for container in ax.containers:
            ax.bar_label(container)
            
        plt.tight_layout()
        plt.show()


# In[7]:


analyze_missingness(df)


# In[9]:


demo_cols = ['Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']
print("--- Plotting Demographics ---")
plot_distributions(df, demo_cols, title_prefix="Demographic")


# In[12]:


question_cols = [
            'comma_preference', 
            'heard_of_comma', 
            'care_oxford_comma', 
            'grammar_importance'
        ]
print("--- Plotting Substantive Answers ---")
plot_distributions(df, question_cols, title_prefix="Response")
        
# Specific print out for the main Oxford Comma question
if 'comma_preference' in df.columns:
    print("--- Oxford Comma Preference Breakdown ---")
    print(df['comma_preference'].value_counts(normalize=True) * 100)

