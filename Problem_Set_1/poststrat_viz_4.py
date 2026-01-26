#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#paths
POST_HUMAN_PATH = "Problem_Set_1/poststrat_survey_bin_probs.csv"
POST_GPT_PATH   = "Problem_Set_1/poststrat_gpt_bin_probs.csv"
POST_DIFF_PATH  = "Problem_Set_1/poststrat_gpt_minus_human_bin_probs.csv"

HUMAN_RAW_PATH  = "Problem_Set_1/comma-survey.csv"
GPT_RAW_PATH    = "Problem_Set_1/gpt_comma_survey.csv"

OUTDIR = "Problem_Set_1/data"  # where figures will be saved

#column mapping 
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

demo_cols = ['Gender', 'Age', 'Household Income', 'Education', 'Location (Census Region)']

question_cols = [
    'comma_preference',
    'heard_of_comma',
    'care_oxford_comma',
    'sentence_preference',
    'data_singular_plural_consideration',
    'care_data_debate',
    'grammar_importance'
]

q_map = {f"q{i}": col for i, col in enumerate(question_cols, start=1)}

#valid responses + fuzzy matcher
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
    if pd.isna(response):
        return response
    cleaned = str(response).lower()

    if col == 'comma_preference':
        if 'kind, and' in cleaned:
            return "It's important for a person to be honest, kind, and loyal."
        elif 'kind and' in cleaned:
            return "It's important for a person to be honest, kind and loyal."

    if col == 'sentence_preference':
        if 'data are' in cleaned:
            return "Some experts say it's important to drink milk, but the data are inconclusive."
        elif 'data is' in cleaned:
            return "Some experts say it's important to drink milk, but the data is inconclusive."

    cleaned_simple = cleaned.strip().strip('"').strip('.').strip()
    for valid in valid_options:
        if valid.lower() == cleaned_simple:
            return valid
    return response

#helper for raw distributions
def load_and_clean_human(path):
    df = pd.read_csv(path)
    df.rename(columns=column_mapping, inplace=True)

    df_clean = df.dropna(subset=demo_cols + question_cols).copy()
    # Normalize question responses (trim)
    for c in question_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()
    return df_clean

def load_and_clean_gpt(path):
    df = pd.read_csv(path)
    df.rename(columns=column_mapping, inplace=True)

    df_clean = df.dropna(subset=demo_cols + question_cols).copy()

    #fuzzy-match GPT answers into valid options per question
    for col in question_cols:
        df_clean[col] = df_clean[col].apply(lambda x: match_to_valid_fuzzy(x, valid_responses[col], col))

    #trim
    for c in question_cols:
        df_clean[c] = df_clean[c].astype(str).str.strip()
    return df_clean

def raw_prob_table(df_clean):
    #returns dict: question_col -> Series(probabilities indexed by response label)
    out = {}
    for col in question_cols:
        vc = df_clean[col].value_counts(dropna=False)
        probs = (vc / vc.sum()).sort_index()
        out[col] = probs
    return out

#helper to load poststrat tables
def load_poststrat(path):
    #saved with index=True, so first column is bin label
    #after reading, set as index
    df = pd.read_csv(path)
    #ff saved by pandas with index, will be in a column named something like "Unnamed: 0"
    idx_col = df.columns[0]
    df = df.set_index(idx_col)
    #ensure numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df

post_h = load_poststrat(POST_HUMAN_PATH)      # rows=bins, cols=q1..q7
post_g = load_poststrat(POST_GPT_PATH)
post_d = load_poststrat(POST_DIFF_PATH)       # gpt - human

#build RAW probability tables in the same "bins x q#" shape
human_clean = load_and_clean_human(HUMAN_RAW_PATH)
gpt_clean   = load_and_clean_gpt(GPT_RAW_PATH)

raw_h_dict = raw_prob_table(human_clean)
raw_g_dict = raw_prob_table(gpt_clean)

def dict_to_bins_by_q(raw_dict):
    #raw_dict: question_col -> Series(index=response_label, value=prob)
    #return DataFrame(index=bins, columns=q1..q7)
    series_by_q = {}
    for qname, col in q_map.items():
        s = raw_dict[col].copy()
        series_by_q[qname] = s
    df = pd.DataFrame(series_by_q).fillna(0.0)
    return df

raw_h = dict_to_bins_by_q(raw_h_dict)
raw_g = dict_to_bins_by_q(raw_g_dict)

#align bin indices across all tables
all_bins = sorted(set(post_h.index) | set(post_g.index) | set(raw_h.index) | set(raw_g.index))
post_h = post_h.reindex(all_bins).fillna(0.0)
post_g = post_g.reindex(all_bins).fillna(0.0)
post_d = post_d.reindex(all_bins).fillna(0.0)
raw_h  = raw_h.reindex(all_bins).fillna(0.0)
raw_g  = raw_g.reindex(all_bins).fillna(0.0)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.makedirs(OUTDIR, exist_ok=True)

#helpers
def abbrev_5_words(label: str) -> str:
    words = str(label).split()
    return " ".join(words[:5]) + ("…" if len(words) > 5 else "")

def save_fig(fig, filename):
    path = f"{OUTDIR}/{filename}"
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("Saved:", path)

def normalize_to_1(s: pd.Series) -> pd.Series:
    s = s.astype(float).fillna(0.0)
    total = s.sum()
    if total <= 0:
        return s  # keep zeros
    return s / total

def plot_100pct_stacked(ax, series_dict, title, ylabel="Share"):
    """
    series_dict: dict {bar_label: pd.Series(index=bins, values=prob)}
    Draws 100% stacked bars with bins as segments.
    """
    bar_labels = list(series_dict.keys())
    bins = series_dict[bar_labels[0]].index.tolist()

    #build matrix (n_bars x n_bins)
    mat = np.vstack([series_dict[bl].reindex(bins).to_numpy(dtype=float) for bl in bar_labels])

    #stack
    bottom = np.zeros(len(bar_labels))
    x = np.arange(len(bar_labels))

    for j, b in enumerate(bins):
        vals = mat[:, j]
        ax.bar(x, vals, bottom=bottom, label=b)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    #legend outside (better when many bins)
    ax.legend(
        title="Answer",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

def get_nonzero_bins(*series_list, eps=0.0):
    """
    Keep bins where any of the provided series is nonzero.
    """
    bins = series_list[0].index
    m = np.zeros(len(bins), dtype=bool)
    for s in series_list:
        m |= (s.reindex(bins).abs().to_numpy() > eps)
    return bins[m]

#names for questions
q_pretty = {
    "q1": "Comma preference",
    "q2": "Heard of Oxford comma",
    "q3": "Care about Oxford comma",
    "q4": "Sentence preference",
    "q5": "Considered 'data' singular/plural",
    "q6": "Care about 'data' debate",
    "q7": "Grammar importance",
}

def short_bins(index):
    return [abbrev_5_words(x) for x in index]

#ensure all tables aligned already from earlier code:
#post_h, post_g, raw_h, raw_g exist, share same indices/columns
qs = list(post_h.columns)

# Grouped 100% stacked bars (ALL QUESTIONS ON ONE AXIS)

def plot_grouped_100pct_stacked_all_questions(
    title,
    left_label,
    right_label,
    left_df,
    right_df,
    filename,
    eps=0.0,
    figsize=(14, 6)
):
    qs = list(left_df.columns)
    x = np.arange(len(qs))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    #collect FULL answer labels (nonzero only), per question
    all_full_answers = []
    per_q_bins = {}  # q -> Index of full answer labels kept
    for q in qs:
        bins_keep = get_nonzero_bins(left_df[q], right_df[q], eps=eps)  # this returns Index from df.index
        per_q_bins[q] = bins_keep
        all_full_answers.extend(list(bins_keep))

    #stable unique full labels
    all_full_answers = list(dict.fromkeys(all_full_answers))

    #build display labels (abbrev), but disambiguate collisions
    base_disp = {full: abbrev_5_words(full) for full in all_full_answers}
    disp_counts = {}
    disp_label = {}

    for full in all_full_answers:
        d = base_disp[full]
        disp_counts[d] = disp_counts.get(d, 0) + 1
        # provisional
        disp_label[full] = d

    #if collisions exist, append (1), (2), ... etc
    collision_indices = {}
    for full in all_full_answers:
        d = base_disp[full]
        if disp_counts[d] > 1:
            collision_indices[d] = collision_indices.get(d, 0) + 1
            disp_label[full] = f"{d} ({collision_indices[d]})"

    #assign fixed colors per FULL answer label
    cmap = plt.get_cmap("tab20")
    color_map = {full: cmap(i % cmap.N) for i, full in enumerate(all_full_answers)}

    legend_handles = {}

    #plot each question: two grouped 100% stacked bars
    for i, q in enumerate(qs):
        bins_keep = per_q_bins[q]  # full labels
        left = normalize_to_1(left_df.loc[bins_keep, q].copy())
        right = normalize_to_1(right_df.loc[bins_keep, q].copy())

        bottom_left = 0.0
        bottom_right = 0.0

        for full in bins_keep:
            lv = float(left.get(full, 0.0))
            rv = float(right.get(full, 0.0))
            c = color_map[full]
            lab = disp_label[full]

            h1 = ax.bar(
                i - width/2, lv, width,
                bottom=bottom_left,
                color=c,
                edgecolor="white",
                label=lab
            )
            ax.bar(
                i + width/2, rv, width,
                bottom=bottom_right,
                color=c,
                edgecolor="white"
            )

            bottom_left += lv
            bottom_right += rv

            #save one legend handle per display label
            if lab not in legend_handles:
                legend_handles[lab] = h1

    ax.set_xticks(x)
    ax.set_xticklabels([q_pretty.get(q, q) for q in qs], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_title(title)

    #global legend
    ax.legend(
        legend_handles.values(),
        legend_handles.keys(),
        title="Answer",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    save_fig(fig, filename)

    #human vs GPT (post-stratified)
plot_grouped_100pct_stacked_all_questions(
    title="Post-stratified distributions: Human vs GPT",
    left_label="Human (post-strat)",
    right_label="GPT (post-strat)",
    left_df=post_h,
    right_df=post_g,
    filename="ONEPLOT_poststrat_human_vs_gpt.png"
)

#human raw vs Human post-strat
plot_grouped_100pct_stacked_all_questions(
    title="Human distributions: Raw vs Post-stratified",
    left_label="Human (raw)",
    right_label="Human (post-strat)",
    left_df=raw_h,
    right_df=post_h,
    filename="ONEPLOT_human_raw_vs_poststrat.png"
)

#GPT raw vs GPT post-strat
plot_grouped_100pct_stacked_all_questions(
    title="GPT distributions: Raw vs Post-stratified",
    left_label="GPT (raw)",
    right_label="GPT (post-strat)",
    left_df=raw_g,
    right_df=post_g,
    filename="ONEPLOT_gpt_raw_vs_poststrat.png"
)

#human raw vs GPT raw
plot_grouped_100pct_stacked_all_questions(
    title="Raw distributions: Human vs GPT",
    left_label="Human (raw)",
    right_label="GPT (raw)",
    left_df=raw_h,
    right_df=raw_g,
    filename="ONEPLOT_raw_human_vs_gpt.png"
)

print("\nDone. All questions are combined into single 100% stacked plots.")