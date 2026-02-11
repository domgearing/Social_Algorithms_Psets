#!/usr/bin/env python3
# %% 
from bluesky_helpers import(
    load_name_data, infer_gender, load_json, load_senators
)
# for reading in json files
import os 

# %%
# 1. Load SSA name data
name_data = load_name_data()

# 2. Load senator info (so we know each senator's gender)
senators = load_senators('senators_bluesky.csv')
senator_gender = {s['handle']: s['gender'] for s in senators}

# 3. Load all reply JSON files and run gender inference
results = {}  # senator_handle -> list of inferred genders

reply_files = [f for f in os.listdir('.') if f.startswith('replies_') and f.endswith('.json')]

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
print(f"  Female:       {classified_f} ({classified_f/total_repliers*100:.1f}%)")
print(f"  Male:         {classified_m} ({classified_m/total_repliers*100:.1f}%)")
print(f"  Unknown:      {classified_u} ({classified_u/total_repliers*100:.1f}%)")


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
## visualization of homophily results
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(2)  # two groups: Female Senators, Male Senators
width = 0.35

# Female senators' replier breakdown
fem_sen_bars = [0.380, 1 - 0.380]  # [female repliers, male repliers]
# Male senators' replier breakdown
male_sen_bars = [counts['M']['female_repliers'] / m_total, counts['M']['male_repliers'] / m_total]

bars1 = ax.bar(x - width/2, [fem_sen_bars[0], male_sen_bars[0]], width, label='Female Repliers', color='salmon')
bars2 = ax.bar(x + width/2, [fem_sen_bars[1], male_sen_bars[1]], width, label='Male Repliers', color='steelblue')

# Add baseline line
ax.axhline(y=p_female, color='salmon', linestyle='--', alpha=0.7, label=f'Baseline female ({p_female:.3f})')
ax.axhline(y=p_male, color='steelblue', linestyle='--', alpha=0.7, label=f'Baseline male ({p_male:.3f})')

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
from scipy.stats import chi2_contingency
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
# %%
