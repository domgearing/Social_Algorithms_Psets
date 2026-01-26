"""
compute bin probabilities for raw survey and GPT data before poststratification using functions from survey_poststrat.py.
"""


import pandas as pd
from survey_poststrat import label_encoders, label_encoders_gpt, demo_cols, question_cols, match_to_valid_fuzzy

# Column mapping for renaming
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

# Load, rename raw survey data
df = pd.read_csv('Problem_Set_1/comma-survey.csv')
df.rename(columns=column_mapping, inplace=True)

# Load, rename GPT survey data
df_gpt = pd.read_csv('Problem_Set_1/gpt_comma_survey.csv')
df_gpt.rename(columns=column_mapping, inplace=True)

#valid responses for fuzzy matching (copied from survey_poststrat.py)
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

# Apply fuzzy matching to each question column in GPT survey
for col in question_cols:
    df_gpt[col] = df_gpt[col].apply(lambda x: match_to_valid_fuzzy(x, valid_responses[col], col))

def get_raw_bin_probs(df, encoders, demo_cols, question_cols, prefix):
    out = {}
    for qi, col in enumerate(question_cols, 1):
        le_key = f"y{qi}_{prefix}"
        le = encoders[le_key]
        y = le.transform(df[col].dropna())
        # Bin probabilities: value counts normalized
        bin_probs = pd.Series(y).value_counts(normalize=True).sort_index()
        # Map back to original labels
        labels = le.inverse_transform(bin_probs.index)
        out[f"q{qi}"] = pd.Series(bin_probs.values, index=labels)
    return pd.DataFrame(out).fillna(0.0)

# Compute for survey and GPT
df_clean = df.dropna(subset=demo_cols + question_cols)
df_gpt_clean = df_gpt.dropna(subset=demo_cols + question_cols)

survey_bin_probs = get_raw_bin_probs(df_clean, label_encoders, demo_cols, question_cols, "survey")
gpt_bin_probs = get_raw_bin_probs(df_gpt_clean, label_encoders_gpt, demo_cols, question_cols, "gpt_survey")

print("Raw Survey Bin Probabilities:")
print(survey_bin_probs)
print("\nGPT Survey Bin Probabilities:")
print(gpt_bin_probs)

survey_bin_probs.to_csv("raw_survey_bin_probs.csv")
gpt_bin_probs.to_csv("raw_gpt_bin_probs.csv")
print("\nSaved to raw_survey_bin_probs.csv and raw_gpt_bin_probs.csv")