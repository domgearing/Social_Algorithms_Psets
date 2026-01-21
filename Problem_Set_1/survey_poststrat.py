# In[1]:
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder


# In[2]:
# Load the CSV file into a DataFrame
df = pd.read_csv('comma-survey.csv')
# Rename columns for easier access
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

# In[3]:
demo_cols = ['Gender', 'Age', 
             'Household Income', 
             'Education', 
             'Location (Census Region)']

question_cols = [
            'comma_preference', 
            'heard_of_comma', 
            'care_oxford_comma', 
            'sentence_preference',
            'data_singular_plural_consideration',
            'care_data_debate',
            'grammar_importance'
        ]