#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[2]:
# Load the CSV file into a DataFrame
df = pd.read_csv('Problem_Set_1/comma-survey.csv')
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

df_clean = df.dropna(subset=demo_cols + question_cols)

print(f"Original rows: {len(df)}")
print(f"After dropping missing: {len(df_clean)}")

# In[4]:
X = df_clean[demo_cols]

# One-hot encode the demographic variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)


# %%

def encode(series, name):
    le = LabelEncoder()
    y = le.fit_transform(series)
    label_encoders[name] = le
    return y

label_encoders = {}

encode(df_clean['comma_preference'], 'y1_survey')
encode(df_clean['heard_of_comma'], 'y2_survey')
encode(df_clean['care_oxford_comma'], 'y3_survey')
encode(df_clean['sentence_preference'], 'y4_survey')
encode(df_clean['data_singular_plural_consideration'], 'y5_survey')
encode(df_clean['care_data_debate'], 'y6_survey')
encode(df_clean['grammar_importance'], 'y7_survey')

# %%
# fitting the multinomial logistic regression models

from sklearn.base import clone

models_survey = {}

#pap question names to corresponding 1D encoded arrays
ys_survey = {
    "q1": label_encoders['y1_survey'].transform(df_clean['comma_preference']),
    "q2": label_encoders['y2_survey'].transform(df_clean['heard_of_comma']),
    "q3": label_encoders['y3_survey'].transform(df_clean['care_oxford_comma']),
    "q4": label_encoders['y4_survey'].transform(df_clean['sentence_preference']),
    "q5": label_encoders['y5_survey'].transform(df_clean['data_singular_plural_consideration']),
    "q6": label_encoders['y6_survey'].transform(df_clean['care_data_debate']),
    "q7": label_encoders['y7_survey'].transform(df_clean['grammar_importance'])
}

lr = LogisticRegression(solver='lbfgs', max_iter=1000)
#have to clone so you get a new lr object each time
#  instead of overwriting
for name, y in ys_survey.items():
    m = clone(lr)             
    models_survey[name] = m.fit(X_encoded, y)



# In[5]:
#repeat for gpt_comma_survey.csv
df_gpt = pd.read_csv('Problem_Set_1/gpt_comma_survey.csv')
df_gpt.rename(columns=column_mapping, inplace=True)


df_gpt_clean = df_gpt.dropna(subset=demo_cols + question_cols)
print(f"GPT Original rows: {len(df_gpt)}")
print(f"GPT After dropping missing: {len(df_gpt_clean)}")

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
    cleaned_simple = cleaned.strip().strip('"').strip('.').strip()
    for valid in valid_options:
        if valid.lower() == cleaned_simple:
            return valid
    
    print(f"No match found for: {repr(response)}")
    return response

# Apply the mapping to each column
for col in question_cols:
    df_gpt_clean[col] = df_gpt_clean[col].apply(
        lambda x: match_to_valid_fuzzy(x, valid_responses[col], col)
    )

# %%
X_gpt = df_gpt_clean[demo_cols]
# One-hot encode the demographic variables
X_gpt_encoded = encoder.transform(X_gpt)

def encode_gpt(series, name):
    le = LabelEncoder()
    y = le.fit_transform(series)
    label_encoders_gpt[name] = le
    return y

label_encoders_gpt = {}

encode_gpt(df_gpt_clean['comma_preference'], 'y1_gpt_survey')
encode_gpt(df_gpt_clean['heard_of_comma'], 'y2_gpt_survey')
encode_gpt(df_gpt_clean['care_oxford_comma'], 'y3_gpt_survey')
encode_gpt(df_gpt_clean['sentence_preference'], 'y4_gpt_survey')
encode_gpt(df_gpt_clean['data_singular_plural_consideration'], 'y5_gpt_survey')
encode_gpt(df_gpt_clean['care_data_debate'], 'y6_gpt_survey')
encode_gpt(df_gpt_clean['grammar_importance'], 'y7_gpt_survey')

# %%
# fitting the multinomial logistic regression models
models_gpt_survey = {}

#map question names to corresponding 1D encoded arrays for GPT survey
ys_gpt_survey = {
    "q1": label_encoders_gpt['y1_gpt_survey'].transform(df_gpt_clean['comma_preference']),
    "q2": label_encoders_gpt['y2_gpt_survey'].transform(df_gpt_clean['heard_of_comma']),
    "q3": label_encoders_gpt['y3_gpt_survey'].transform(df_gpt_clean['care_oxford_comma']),
    "q4": label_encoders_gpt['y4_gpt_survey'].transform(df_gpt_clean['sentence_preference']),
    "q5": label_encoders_gpt['y5_gpt_survey'].transform(df_gpt_clean['data_singular_plural_consideration']),
    "q6": label_encoders_gpt['y6_gpt_survey'].transform(df_gpt_clean['care_data_debate']),
    "q7": label_encoders_gpt['y7_gpt_survey'].transform(df_gpt_clean['grammar_importance'])
}

lr = LogisticRegression(solver='lbfgs', max_iter=1000)
#have to clone so you get a new lr object each time
#  instead of overwriting
for name, y in ys_gpt_survey.items():
    m = clone(lr)             
    models_gpt_survey[name] = m.fit(X_gpt_encoded, y)

# %%


import pandas as pd
import numpy as np

#load census post-strat cell table
CENSUS_CELLS_PATH = "Problem_Set_1/post_strat_long_full_cartesian_FIXED_v2.csv"  # <- adjust if needed
cells = pd.read_csv(CENSUS_CELLS_PATH)

# EXPECTED columns in cells:
#   census_region, age_group, sex, education_5, income_bin, pop_count
# map them into the survey's demo_cols:
#   'Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)'

# Map census columns -> survey demographic column names
cells_for_encoder = pd.DataFrame({
    "Gender": cells["sex"].astype(str).str.strip(),
    "Age": cells["age_group"].astype(str).str.strip(),
    "Household Income": cells["income_bin"].astype(str).str.strip(),
    "Education": cells["education_5"].astype(str).str.strip(),
    "Location (Census Region)": cells["census_region"].astype(str).str.strip(),
})

#transform census cells using SAME encoder fit on survey data
X_cells_encoded = encoder.transform(cells_for_encoder)

#weights
w = cells["pop_count"].to_numpy(dtype=float)
w_sum = w.sum()
if w_sum == 0:
    raise ValueError("pop_count sums to 0 in census cells table. Something is wrong upstream.")

#helper: post-stratified class probabilities for one model
def poststrat_probs(model, X_cells_encoded, weights):
    """
    Returns:
      weighted_probs: array (n_classes,)
    """
    probs = model.predict_proba(X_cells_encoded)         # (n_cells, n_classes)
    wp = (probs * weights[:, None]).sum(axis=0) / weights.sum()
    return wp

def poststrat_all_questions(models_dict, label_encoders_dict, prefix):
    """
    models_dict: {"q1": fitted_model, ..., "q7": fitted_model}
    label_encoders_dict: dict that has LabelEncoder objects stored under keys like 'y1_survey' or 'y1_gpt_survey'
    prefix: either "survey" or "gpt", used to pick right encoders
    """
    out = {}

    for qi in range(1, 8):
        qname = f"q{qi}"
        model = models_dict[qname]

        #get corresponding LabelEncoder 
        #for survey: y1_survey...y7_survey
        #for gpt:    y1_gpt_survey...y7_gpt_survey
        le_key = f"y{qi}_{prefix}"
        le = label_encoders_dict[le_key]

        #weighted probs for each encoded class index
        wp = poststrat_probs(model, X_cells_encoded, w)

        #convert class indices (0..K-1) back to original response labels
        #for sklearn LR, class order is model.classes_ (numeric indices because trained on encoded y)
        class_idx = model.classes_
        class_labels = le.inverse_transform(class_idx)

        out[qname] = pd.Series(wp, index=class_labels)

    #align into single dataframe: rows=bins, cols=questions
    return pd.DataFrame(out).fillna(0.0)

#Post-stratify HUMAN survey models
post_survey = poststrat_all_questions(
    models_dict=models_survey,
    label_encoders_dict=label_encoders,
    prefix="survey"
)

#save + inspect
post_survey.to_csv("poststrat_survey_bin_probs.csv", index=True)
print("\nPost-stratified HUMAN survey bin probabilities saved to poststrat_survey_bin_probs.csv")
print(post_survey)

#Post-stratify GPT survey models
post_gpt = poststrat_all_questions(
    models_dict=models_gpt_survey,
    label_encoders_dict=label_encoders_gpt,
    prefix="gpt_survey"
)

post_gpt.to_csv("poststrat_gpt_bin_probs.csv", index=True)
print("\nPost-stratified GPT survey bin probabilities saved to poststrat_gpt_bin_probs.csv")
print(post_gpt)

#compare GPT vs Human post-strat distributions per bin
# (difference = GPT - Human)
diff = post_gpt.reindex(post_survey.index).fillna(0) - post_survey.fillna(0)
diff.to_csv("poststrat_gpt_minus_human_bin_probs.csv", index=True)
print("\nDifference (GPT - Human) saved to poststrat_gpt_minus_human_bin_probs.csv")