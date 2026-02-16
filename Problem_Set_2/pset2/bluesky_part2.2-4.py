#!/usr/bin/env python3
# %%
from bluesky_helpers import (
    load_name_data, infer_gender, load_json, load_senators, parse_datetime
)
# for reading in json files
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency


# %%
# 1. Load SSA name data
name_data = load_name_data()

# 2. Load senator info (so we know each senator's gender)
senators = load_senators('senators_bluesky.csv')
senator_gender = {s['handle']: s['gender'] for s in senators}

# 3. Load all reply JSON files and run gender inference
results = {}  # senator_handle -> list of inferred genders

reply_files = [f for f in os.listdir('.') if f.startswith(
    'replies_') and f.endswith('.json')]

total_repliers = 0
classified_f = 0
classified_m = 0
classified_u = 0

for filename in reply_files:
    data = load_json(filename)

    for post in data:
        for reply in post['replies']:
            display_name = reply.get('displayName', '')
            gender = infer_gender(display_name, name_data)
            reply['inferred_gender'] = gender  # store it for later use

            total_repliers += 1
            if gender == 'F':
                classified_f += 1
            elif gender == 'M':
                classified_m += 1
            else:
                classified_u += 1

# 4. Report classification results
classified = classified_f + classified_m
print(f"Total repliers: {total_repliers}")
print(f"Classified:     {classified} ({classified/total_repliers*100:.1f}%)")
print(
    f"  Female:       {classified_f} ({classified_f/total_repliers*100:.1f}%)")
print(
    f"  Male:         {classified_m} ({classified_m/total_repliers*100:.1f}%)")
print(
    f"  Unknown:      {classified_u} ({classified_u/total_repliers*100:.1f}%)")


# %%
# ==========================================
# II.3 Homophily Measurement
# ==========================================

# Count replier genders split by senator gender
counts = {
    'F': {'female_repliers': 0, 'male_repliers': 0},
    'M': {'female_repliers': 0, 'male_repliers': 0},
}

for senator in senators:
    handle = senator['handle']
    filename = f"replies_{handle.replace('.', '_')}.json"

    try:
        data = load_json(filename)
    except FileNotFoundError:
        continue

    sen_gender = senator['gender']

    for post in data:
        for reply in post['replies']:
            display_name = reply.get('displayName', '')
            g = infer_gender(display_name, name_data)

            if g == 'F':
                counts[sen_gender]['female_repliers'] += 1
            elif g == 'M':
                counts[sen_gender]['male_repliers'] += 1

# Baseline
p_female = classified_f / classified
p_male = classified_m / classified

print(f"\n--- Homophily Analysis ---")
print(f"Baseline: p_female = {p_female:.3f}, p_male = {p_male:.3f}")

# Observed rates
f_total = counts['F']['female_repliers'] + counts['F']['male_repliers']
m_total = counts['M']['female_repliers'] + counts['M']['male_repliers']

obs_F = counts['F']['female_repliers'] / f_total
obs_M = counts['M']['male_repliers'] / m_total

print(f"Female senators: {obs_F:.3f} of repliers are female")
print(f"Male senators:   {obs_M:.3f} of repliers are male")

# Homophily coefficients
H_female = obs_F - p_female
H_male = obs_M - p_male

print(f"\nH_female = {H_female:+.3f}")
print(f"H_male   = {H_male:+.3f}")


# %%
# visualization of homophily results


fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(2)  # two groups: Female Senators, Male Senators
width = 0.35

# Female senators' replier breakdown
fem_sen_bars = [0.380, 1 - 0.380]  # [female repliers, male repliers]
# Male senators' replier breakdown
male_sen_bars = [counts['M']['female_repliers'] /
                 m_total, counts['M']['male_repliers'] / m_total]

bars1 = ax.bar(x - width/2, [fem_sen_bars[0], male_sen_bars[0]],
               width, label='Female Repliers', color='salmon')
bars2 = ax.bar(x + width/2, [fem_sen_bars[1], male_sen_bars[1]],
               width, label='Male Repliers', color='steelblue')

# Add baseline line
ax.axhline(y=p_female, color='salmon', linestyle='--',
           alpha=0.7, label=f'Baseline female ({p_female:.3f})')
ax.axhline(y=p_male, color='steelblue', linestyle='--',
           alpha=0.7, label=f'Baseline male ({p_male:.3f})')

ax.set_xticks(x)
ax.set_xticklabels(['Female Senators', 'Male Senators'])
ax.set_ylabel('Proportion of Repliers')
ax.set_title('Gender Breakdown of Repliers by Senator Gender')
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('homophily_visualization.png', dpi=300, bbox_inches='tight')
plt.show()


# %%
# Statistical Test for homophily significance

# Construct contingency table
#             | Female Repliers | Male Repliers
# Female Senators |      a      |      b
# Male Senators   |      c      |      d

contingency_table = [
    [counts['F']['female_repliers'], counts['F']['male_repliers']],
    [counts['M']['female_repliers'], counts['M']['male_repliers']]
]

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square test for homophily significance:")
print(f"Chi-square statistic: {chi2:.3f}")
print(f"P-value: {p_value:.3f}")
if p_value < 0.05:
    print("The homophily is statistically significant.")
else:
    print("The homophily is not statistically significant.")

# Visualize observed vs expected counts under independence
observed = np.array(contingency_table)
expected = np.array(expected)
labels = [
    "Female senators -> Female repliers",
    "Female senators -> Male repliers",
    "Male senators -> Female repliers",
    "Male senators -> Male repliers",
]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(labels))
width = 0.35

ax.bar(x - width / 2, observed.flatten(), width,
       label="Observed", color="slateblue")
ax.bar(x + width / 2, expected.flatten(), width,
       label="Expected (independence)", color="lightgray")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("Reply counts")
ax.set_title("Observed vs Expected Reply Counts (Chi-square Test)")
ax.legend()

plt.tight_layout()
plt.savefig("chi_square_observed_expected.png", dpi=300, bbox_inches="tight")
plt.show()


# %%
# II.4 Reply Timing Analysis
qualifying_posts = []
high_reply_senators = {}  # track who gets 200+ replies


for senator in senators:
    handle = senator['handle']
    filename = f"replies_{handle.replace('.', '_')}.json"
    try:
        data = load_json(filename)
    except FileNotFoundError:
        continue

    for post in data:
        rc = post['replyCount']
        if 50 <= rc <= 200:
            qualifying_posts.append(post)
        if rc > 200:
            if handle not in high_reply_senators:
                high_reply_senators[handle] = 0
            high_reply_senators[handle] += 1

print(f"Posts with 50-200 replies: {len(qualifying_posts)}")
print(f"\nSenators with 200+ reply posts:")
for handle, count in sorted(high_reply_senators.items(), key=lambda x: -x[1]):
    print(f"  {handle}: {count} post(s)")

# %%
# 2. Split replies into early 25% and late 25%, compare
early_genders = {'F': 0, 'M': 0, 'U': 0}
late_genders = {'F': 0, 'M': 0, 'U': 0}
early_lengths = []
late_lengths = []
early_likes = []
late_likes = []

for post in qualifying_posts:
    replies = post['replies']

    # Sort by timestamp
    sorted_replies = sorted(replies, key=lambda r: r.get('createdAt', ''))

    n = len(sorted_replies)
    cutoff_25 = n // 4  # first 25%
    cutoff_75 = n - (n // 4)  # last 25%

    early = sorted_replies[:cutoff_25]
    late = sorted_replies[cutoff_75:]

    for reply in early:
        g = infer_gender(reply.get('displayName', ''), name_data)
        early_genders[g] += 1
        early_lengths.append(len(reply.get('text', '')))
        early_likes.append(reply.get('likeCount', 0))

    for reply in late:
        g = infer_gender(reply.get('displayName', ''), name_data)
        late_genders[g] += 1
        late_lengths.append(len(reply.get('text', '')))
        late_likes.append(reply.get('likeCount', 0))

# %%
# 3. Report results
early_classified = early_genders['F'] + early_genders['M']
late_classified = late_genders['F'] + late_genders['M']

print(f"\n--- Early vs Late Reply Comparison ---")
print(f"Early replies (first 25%): {early_classified} classified")
print(f"  Female: {early_genders['F']/early_classified:.3f}")
print(f"  Male:   {early_genders['M']/early_classified:.3f}")

print(f"Late replies (last 25%):  {late_classified} classified")
print(f"  Female: {late_genders['F']/late_classified:.3f}")
print(f"  Male:   {late_genders['M']/late_classified:.3f}")

print(
    f"\nAvg reply length - Early: {sum(early_lengths)/len(early_lengths):.1f} chars, Late: {sum(late_lengths)/len(late_lengths):.1f} chars")
print(
    f"Avg likes - Early: {sum(early_likes)/len(early_likes):.2f}, Late: {sum(late_likes)/len(late_likes):.2f}")

# %%

# Exploratory visualizations for reply timing analysis

# 1) Gender composition: early vs late (classified only)
fig, ax = plt.subplots(figsize=(6, 4))
early_f = early_genders['F'] / early_classified if early_classified else 0
early_m = early_genders['M'] / early_classified if early_classified else 0
late_f = late_genders['F'] / late_classified if late_classified else 0
late_m = late_genders['M'] / late_classified if late_classified else 0

ax.bar([0, 1], [early_f, late_f], label='Female', color='salmon')
ax.bar([0, 1], [early_m, late_m], bottom=[
       early_f, late_f], label='Male', color='steelblue')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Early (first 25%)', 'Late (last 25%)'])
ax.set_ylabel('Proportion of classified repliers')
ax.set_title('Gender Composition: Early vs Late Replies')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('reply_timing_gender_composition.png',
            dpi=300, bbox_inches='tight')
plt.show()

# 2) Reply length distribution: early vs late
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([early_lengths, late_lengths], labels=[
           'Early', 'Late'], showfliers=False)
ax.set_ylabel('Reply length (chars)')
ax.set_title('Reply Length: Early vs Late')
plt.tight_layout()
plt.savefig('reply_timing_length_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 3) Likes distribution: early vs late
fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot([early_likes, late_likes], labels=[
           'Early', 'Late'], showfliers=False)
ax.set_ylabel('Likes')
ax.set_title('Reply Likes: Early vs Late')
plt.tight_layout()
plt.savefig('reply_timing_likes_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 4) Senators with 200+ reply posts
if high_reply_senators:
    handle_to_name = {s['handle']: s['name'] for s in senators}
    handles = [k for k, _ in sorted(
        high_reply_senators.items(), key=lambda x: -x[1])]
    labels = [handle_to_name.get(h, h) for h in handles]
    counts = [high_reply_senators[h] for h in handles]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, counts, color='darkorange')
    ax.set_ylabel('Posts with 200+ replies')
    ax.set_title('High-Reply Posts by Senator')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('high_reply_posts_by_senator.png',
                dpi=300, bbox_inches='tight')
    plt.show()

# %%
# Time-bin visualizations (quartiles)
bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
n_bins = len(bin_labels)

bin_gender_counts = [{'F': 0, 'M': 0, 'U': 0} for _ in range(n_bins)]
bin_lengths = [[] for _ in range(n_bins)]
bin_likes = [[] for _ in range(n_bins)]

for post in qualifying_posts:
    replies = post['replies']
    sorted_replies = sorted(replies, key=lambda r: r.get('createdAt', ''))
    n = len(sorted_replies)
    if n == 0:
        continue

    for idx, reply in enumerate(sorted_replies):
        frac = (idx + 0.5) / n
        bin_index = min(int(frac * n_bins), n_bins - 1)

        g = infer_gender(reply.get('displayName', ''), name_data)
        bin_gender_counts[bin_index][g] += 1
        bin_lengths[bin_index].append(len(reply.get('text', '')))
        bin_likes[bin_index].append(reply.get('likeCount', 0))

# 1) Gender composition across bins (classified only)
fig, ax = plt.subplots(figsize=(7, 4))
female_props = []
male_props = []

for counts in bin_gender_counts:
    classified = counts['F'] + counts['M']
    if classified == 0:
        female_props.append(0)
        male_props.append(0)
    else:
        female_props.append(counts['F'] / classified)
        male_props.append(counts['M'] / classified)

x = np.arange(n_bins)
ax.bar(x, female_props, label='Female', color='salmon')
ax.bar(x, male_props, bottom=female_props, label='Male', color='steelblue')
ax.set_xticks(x)
ax.set_xticklabels(bin_labels)
ax.set_ylabel('Proportion of classified repliers')
ax.set_title('Gender Composition by Reply Time Bin')
ax.legend()
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig('reply_timing_gender_bins.png', dpi=300, bbox_inches='tight')
plt.show()

# 2) Reply length by time bin
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(bin_lengths, labels=bin_labels, showfliers=False)
ax.set_ylabel('Reply length (chars)')
ax.set_title('Reply Length by Reply Time Bin')
plt.tight_layout()
plt.savefig('reply_timing_length_bins.png', dpi=300, bbox_inches='tight')
plt.show()

# 3) Likes by time bin
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(bin_likes, labels=bin_labels, showfliers=False)
ax.set_ylabel('Likes')
ax.set_title('Reply Likes by Reply Time Bin')
plt.tight_layout()
plt.savefig('reply_timing_likes_bins.png', dpi=300, bbox_inches='tight')
plt.show()
