
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from datetime import datetime
from survey_analysis import Tee, column_mapping, demo_cols, question_cols

# Create logs and viz folders
os.makedirs('logs', exist_ok=True)
os.makedirs('viz', exist_ok=True)

#setup logging 
log_filename = f"logs/compare_surveys_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file = open(log_filename, 'w')

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_and_prepare_data(filepath):
    """Load CSV and rename columns."""
    df = pd.read_csv(filepath)
    df.rename(columns=column_mapping, inplace=True)
    return df


def compare_missingness_3(df1, df2, df3, name1, name2, name3):
    """Compare missing data between three surveys."""
    print("\n--- Missingness Comparison ---")
    missing1 = (df1.isnull().sum() / len(df1)) * 100
    missing2 = (df2.isnull().sum() / len(df2)) * 100
    missing3 = (df3.isnull().sum() / len(df3)) * 100
    comparison = pd.DataFrame({
        name1: missing1,
        name2: missing2,
        name3: missing3
    })
    print(comparison[(comparison[name1] > 0) | (comparison[name2] > 0) | (comparison[name3] > 0)])


def compare_distributions_grid_3(df1, df2, df3, columns, name1, name2, name3, title_prefix, filename_prefix):
    """Create 100% stacked bar charts for all questions side by side for three surveys."""
    import matplotlib.patches as mpatches
    n_questions = len([col for col in columns if col in df1.columns and col in df2.columns and col in df3.columns])
    # Collect all unique responses across all questions for consistent coloring
    all_responses = set()
    for col in columns:
        if col not in df1.columns or col not in df2.columns or col not in df3.columns:
            continue
        all_responses.update(df1[col].unique())
        all_responses.update(df2[col].unique())
        all_responses.update(df3[col].unique())
    all_responses = sorted([r for r in all_responses if pd.notna(r)])
    # Create color palette for responses
    colors = plt.cm.tab20(range(len(all_responses)))
    response_colors = {resp: colors[i] for i, resp in enumerate(all_responses)}
    # Create single figure with all bars
    fig, ax = plt.subplots(figsize=(24, 7))
    x_pos = 0
    bar_width = 0.25
    group_spacing = 1.0
    question_labels = []
    question_positions = []
    for col_idx, col in enumerate(columns):
        if col not in df1.columns or col not in df2.columns or col not in df3.columns:
            continue
        # Get value counts for all datasets
        counts1 = df1[col].value_counts(normalize=True).sort_index() * 100
        counts2 = df2[col].value_counts(normalize=True).sort_index() * 100
        counts3 = df3[col].value_counts(normalize=True).sort_index() * 100
        # Ensure all responses are represented
        counts1 = counts1.reindex(all_responses, fill_value=0)
        counts2 = counts2.reindex(all_responses, fill_value=0)
        counts3 = counts3.reindex(all_responses, fill_value=0)
        # Store question position for label
        question_positions.append(x_pos + bar_width)
        question_labels.append(col.replace("_", " ").title())
        # Create stacked bars
        bottom1 = 0
        bottom2 = 0
        bottom3 = 0
        for resp in all_responses:
            val1 = counts1.get(resp, 0)
            val2 = counts2.get(resp, 0)
            val3 = counts3.get(resp, 0)
            ax.bar(x_pos, val1, bar_width, bottom=bottom1, color=response_colors[resp], edgecolor='white', linewidth=1)
            ax.bar(x_pos + bar_width, val2, bar_width, bottom=bottom2, color=response_colors[resp], edgecolor='white', linewidth=1)
            ax.bar(x_pos + 2*bar_width, val3, bar_width, bottom=bottom3, color=response_colors[resp], edgecolor='white', linewidth=1)
            # Add percentage labels if > 5%
            if val1 > 5:
                ax.text(x_pos, bottom1 + val1/2, f'{val1:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            if val2 > 5:
                ax.text(x_pos + bar_width, bottom2 + val2/2, f'{val2:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            if val3 > 5:
                ax.text(x_pos + 2*bar_width, bottom3 + val3/2, f'{val3:.0f}%', ha='center', va='center', fontsize=8, fontweight='bold', color='white')
            bottom1 += val1
            bottom2 += val2
            bottom3 += val3
        # Add source labels below bars
        ax.text(x_pos + bar_width/2, -8, name1, ha='center', fontsize=9, fontweight='bold')
        ax.text(x_pos + 1.5*bar_width, -8, name2, ha='center', fontsize=9, fontweight='bold')
        ax.text(x_pos + 2.5*bar_width, -8, name3, ha='center', fontsize=9, fontweight='bold')
        x_pos += group_spacing
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'{title_prefix} Comparison: {name1} vs {name2} vs {name3}', fontsize=14, fontweight='bold')
    ax.set_ylim(-10, 100)
    ax.set_xlim(-0.5, x_pos - 0.3)
    ax.set_xticks(question_positions)
    ax.set_xticklabels(question_labels, fontsize=10, fontweight='bold', rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    # Create legend for responses
    handles = [mpatches.Patch(facecolor=response_colors[resp], edgecolor='white', label=resp) for resp in all_responses]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=10, frameon=True)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    filename = f'comparison_viz_with_gpt_census_demo/{filename_prefix}combined_3surveys.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def compare_distributions_3(df1, df2, df3, columns, name1, name2, name3, title_prefix, filename_prefix):
    """Compare distributions side-by-side for specified columns for three surveys."""
    for idx, col in enumerate(columns):
        if col not in df1.columns or col not in df2.columns or col not in df3.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        # Get value counts as percentages
        counts1 = df1[col].value_counts(normalize=True).sort_index() * 100
        counts2 = df2[col].value_counts(normalize=True).sort_index() * 100
        counts3 = df3[col].value_counts(normalize=True).sort_index() * 100
        all_responses = sorted(set(counts1.index) | set(counts2.index) | set(counts3.index))
        counts1 = counts1.reindex(all_responses, fill_value=0)
        counts2 = counts2.reindex(all_responses, fill_value=0)
        counts3 = counts3.reindex(all_responses, fill_value=0)
        # Plot 1
        axes[0].barh(range(len(counts1)), counts1.values, color='steelblue')
        axes[0].set_yticks(range(len(counts1)))
        axes[0].set_yticklabels(counts1.index, fontsize=9)
        axes[0].set_xlabel('Percentage (%)')
        axes[0].set_title(f'{title_prefix}: {col.replace("_", " ").title()}\n{name1}', fontweight='bold')
        axes[0].set_xlim(0, max(counts1.values.max(), counts2.values.max(), counts3.values.max()) * 1.1)
        for i, v in enumerate(counts1.values):
            axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        # Plot 2
        axes[1].barh(range(len(counts2)), counts2.values, color='coral')
        axes[1].set_yticks(range(len(counts2)))
        axes[1].set_yticklabels(counts2.index, fontsize=9)
        axes[1].set_xlabel('Percentage (%)')
        axes[1].set_title(f'{title_prefix}: {col.replace("_", " ").title()}\n{name2}', fontweight='bold')
        axes[1].set_xlim(0, max(counts1.values.max(), counts2.values.max(), counts3.values.max()) * 1.1)
        for i, v in enumerate(counts2.values):
            axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        # Plot 3
        axes[2].barh(range(len(counts3)), counts3.values, color='seagreen')
        axes[2].set_yticks(range(len(counts3)))
        axes[2].set_yticklabels(counts3.index, fontsize=9)
        axes[2].set_xlabel('Percentage (%)')
        axes[2].set_title(f'{title_prefix}: {col.replace("_", " ").title()}\n{name3}', fontweight='bold')
        axes[2].set_xlim(0, max(counts1.values.max(), counts2.values.max(), counts3.values.max()) * 1.1)
        for i, v in enumerate(counts3.values):
            axes[2].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        plt.tight_layout()
        sanitized_col_name = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f'comparison_viz_with_gpt_census_demo/{filename_prefix}{idx+1:02d}_{sanitized_col_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def main():
    # Get file inputs
    file1 = input("Enter path to first survey CSV: ").strip()
    file2 = input("Enter path to second survey CSV: ").strip()
    file3 = input("Enter path to third survey CSV: ").strip()
    name1 = input("Enter name for first survey (e.g., 'Human'): ").strip()
    name2 = input("Enter name for second survey (e.g., 'GPT'): ").strip()
    name3 = input("Enter name for third survey (e.g., 'GPT Census'): ").strip()
    # Load data
    print(f"\nLoading {file1}...")
    df1 = load_and_prepare_data(file1)
    print(f"Loaded {len(df1)} rows")
    print(f"Loading {file2}...")
    df2 = load_and_prepare_data(file2)
    print(f"Loaded {len(df2)} rows")
    print(f"Loading {file3}...")
    df3 = load_and_prepare_data(file3)
    print(f"Loaded {len(df3)} rows")
    # Compare missingness
    compare_missingness_3(df1, df2, df3, name1, name2, name3)
    # Compare demographics distributions (side by side for all three)
    print(f"\n--- Plotting Demographics Comparison (All 3) ---")
    compare_distributions_3(df1, df2, df3, demo_cols, name1, name2, name3, f"Demographic", f"compare_demo_{name1}_{name2}_{name3}_")
    # Compare question distributions in a combined grid (all three)
    print(f"\n--- Plotting Question Responses Comparison (Combined, 3 Surveys) ---")
    compare_distributions_grid_3(df1, df2, df3, question_cols, name1, name2, name3, "Response", "compare_response_")
    print(f"\nComparison complete! Files saved to comparison_viz_with_gpt_census_demo/ folder.")

if __name__ == "__main__":
    main()
