# In[1]:
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


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

df_clean = df.dropna(subset=demo_cols + question_cols)

print(f"Original rows: {len(df)}")
print(f"After dropping missing: {len(df_clean)}")

# In[4]:
X = df_clean[demo_cols]

# One-hot encode the demographic variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)


# %%
label_encoder = LabelEncoder()
y1_survey = label_encoder.fit_transform(df_clean['comma_preference'])
y2_survey = label_encoder.fit_transform(df_clean['heard_of_comma'])
y3_survey = label_encoder.fit_transform(df_clean['care_oxford_comma'])
y4_survey = label_encoder.fit_transform(df_clean['sentence_preference'])
y5_survey = label_encoder.fit_transform(df_clean['data_singular_plural_consideration'])
y6_survey = label_encoder.fit_transform(df_clean['care_data_debate'])
y7_survey = label_encoder.fit_transform(df_clean['grammar_importance'])

# %%
# fitting the multinomial logistic regression models
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
model1_survey = lr.fit(X_encoded, y1_survey)
model2_survey = lr.fit(X_encoded, y2_survey)
model3_survey = lr.fit(X_encoded, y3_survey)
model4_survey = lr.fit(X_encoded, y4_survey)
model5_survey = lr.fit(X_encoded, y5_survey)
model6_survey = lr.fit(X_encoded, y6_survey)
model7_survey = lr.fit(X_encoded, y7_survey)
# In[5]:



# Now repeat for gpt_comma_survey.csv
df_gpt = pd.read_csv('gpt_comma_survey.csv')
df_gpt.rename(columns=column_mapping, inplace=True)
df_gpt_clean = df_gpt.dropna(subset=demo_cols + question_cols)
print(f"GPT Original rows: {len(df_gpt)}")
print(f"GPT After dropping missing: {len(df_gpt_clean)}")


# %%
X_gpt = df_gpt_clean[demo_cols]
# One-hot encode the demographic variables
X_gpt_encoded = encoder.transform(X_gpt)
y1_gpt = label_encoder.fit_transform(df_gpt_clean['comma_preference'])
y2_gpt = label_encoder.fit_transform(df_gpt_clean['heard_of_comma'])
y3_gpt = label_encoder.fit_transform(df_gpt_clean['care_oxford_comma'])
y4_gpt = label_encoder.fit_transform(df_gpt_clean['sentence_preference'])
y5_gpt = label_encoder.fit_transform(df_gpt_clean['data_singular_plural_consideration'])
y6_gpt = label_encoder.fit_transform(df_gpt_clean['care_data_debate'])
y7_gpt = label_encoder.fit_transform(df_gpt_clean['grammar_importance'])
# %%
# fitting the multinomial logistic regression models
model1_gpt = lr.fit(X_gpt_encoded, y1_gpt)
model2_gpt = lr.fit(X_gpt_encoded, y2_gpt)
model3_gpt = lr.fit(X_gpt_encoded, y3_gpt)
model4_gpt = lr.fit(X_gpt_encoded, y4_gpt)
model5_gpt = lr.fit(X_gpt_encoded, y5_gpt)
model6_gpt = lr.fit(X_gpt_encoded, y6_gpt)
model7_gpt = lr.fit(X_gpt_encoded, y7_gpt)

# In[6]:
print(df_gpt_clean['comma_preference'].value_counts())
# %%
