import pandas as pd
import numpy as np

post_h = pd.read_csv("Problem_Set_1/poststrat_survey_bin_probs.csv", index_col=0)
post_g = pd.read_csv("Problem_Set_1/poststrat_gpt_bin_probs.csv", index_col=0)
raw_h = pd.read_csv("Problem_Set_1/raw_survey_bin_probs.csv", index_col=0)
raw_g = pd.read_csv("Problem_Set_1/raw_gpt_bin_probs.csv", index_col=0)


#pretty question names
q_pretty = {
    "q1": "comma_preference",
    "q2": "heard_of_comma",
    "q3": "care_oxford_comma",
    "q4": "sentence_preference",
    "q5": "data_singular_plural_consideration",
    "q6": "care_data_debate",
    "q7": "grammar_importance",
}

def to_long(df, value_name):
    """bins x questions -> long with columns: answer, question, value"""
    return (df.reset_index()
              .rename(columns={df.index.name or df.reset_index().columns[0]: "answer"})
              .melt(id_vars="answer", var_name="question", value_name=value_name))

#Long-form tables
L_post_h = to_long(post_h, "human_poststrat")
L_post_g = to_long(post_g, "gpt_poststrat")
L_raw_h  = to_long(raw_h,  "human_raw")
L_raw_g  = to_long(raw_g,  "gpt_raw")

#merge all into one long table on (answer, question)
out = (L_post_h.merge(L_post_g, on=["answer", "question"], how="outer")
              .merge(L_raw_h,  on=["answer", "question"], how="outer")
              .merge(L_raw_g,  on=["answer", "question"], how="outer")
       ).fillna(0.0)

#add readable question label
out["question_label"] = out["question"].map(lambda q: q_pretty.get(q, q))

#differences (GPT - Human)
out["diff_poststrat_gpt_minus_human"] = out["gpt_poststrat"] - out["human_poststrat"]
out["diff_raw_gpt_minus_human"]       = out["gpt_raw"] - out["human_raw"]

#change due to post-stratification within each source
out["human_post_minus_raw"] = out["human_poststrat"] - out["human_raw"]
out["gpt_post_minus_raw"]   = out["gpt_poststrat"] - out["gpt_raw"]

#reorder columns nicely
out = out[[
    "question", "question_label", "answer",
    "human_raw", "human_poststrat", "human_post_minus_raw",
    "gpt_raw", "gpt_poststrat", "gpt_post_minus_raw",
    "diff_raw_gpt_minus_human", "diff_poststrat_gpt_minus_human"
]].sort_values(["question", "answer"])

#save
OUT_PATH = "Problem_Set_1/answer_probabilities_raw_poststrat_and_differences.csv"
out.to_csv(OUT_PATH, index=False)
print("Saved:", OUT_PATH)