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

def compare_missingness(df1, df2, name1, name2):
    """Compare missing data between two surveys."""
    print("\n--- Missingness Comparison ---")
    missing1 = (df1.isnull().sum() / len(df1)) * 100
    missing2 = (df2.isnull().sum() / len(df2)) * 100
    
    comparison = pd.DataFrame({
        name1: missing1,
        name2: missing2,
        'Difference': abs(missing1 - missing2)
    })
    print(comparison[comparison['Difference'] > 0].sort_values('Difference', ascending=False))

def compare_distributions_grid(df1, df2, columns, name1, name2, title_prefix, filename_prefix):
    """Create 100% stacked bar charts for all questions side by side."""
    import matplotlib.patches as mpatches
    
    n_questions = len([col for col in columns if col in df1.columns and col in df2.columns])
    
    # Collect all unique responses across all questions for consistent coloring
    all_responses = set()
    for col in columns:
        if col not in df1.columns or col not in df2.columns:
            continue
        all_responses.update(df1[col].unique())
        all_responses.update(df2[col].unique())
    all_responses = sorted([r for r in all_responses if pd.notna(r)])
    
    # Create color palette for responses
    colors = plt.cm.tab20(range(len(all_responses)))
    response_colors = {resp: colors[i] for i, resp in enumerate(all_responses)}
    
    # Create single figure with all bars
    fig, ax = plt.subplots(figsize=(18, 7))
    
    x_pos = 0
    bar_width = 0.35
    group_spacing = 0.8
    question_labels = []
    question_positions = []
    
    for col_idx, col in enumerate(columns):
        if col not in df1.columns or col not in df2.columns:
            continue
        
        # Get value counts for both datasets
        counts1 = df1[col].value_counts(normalize=True).sort_index() * 100
        counts2 = df2[col].value_counts(normalize=True).sort_index() * 100
        
        # Ensure all responses are represented
        counts1 = counts1.reindex(all_responses, fill_value=0)
        counts2 = counts2.reindex(all_responses, fill_value=0)
        
        # Store question position for label
        question_positions.append(x_pos + bar_width / 2)
        question_labels.append(col.replace("_", " ").title())
        
        # Create stacked bars
        bottom1 = 0
        bottom2 = 0
        
        for resp in all_responses:
            val1 = counts1.get(resp, 0)
            val2 = counts2.get(resp, 0)
            
            ax.bar(x_pos, val1, bar_width, bottom=bottom1, 
                   color=response_colors[resp], edgecolor='white', linewidth=1)
            ax.bar(x_pos + bar_width, val2, bar_width, bottom=bottom2, 
                   color=response_colors[resp], edgecolor='white', linewidth=1)
            
            # Add percentage labels if > 5%
            if val1 > 5:
                ax.text(x_pos, bottom1 + val1/2, f'{val1:.0f}%', ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
            if val2 > 5:
                ax.text(x_pos + bar_width, bottom2 + val2/2, f'{val2:.0f}%', ha='center', va='center', 
                       fontsize=8, fontweight='bold', color='white')
            
            bottom1 += val1
            bottom2 += val2
        
        # Add source labels below bars
        ax.text(x_pos + bar_width/4, -8, name1, ha='center', fontsize=9, fontweight='bold')
        ax.text(x_pos + 3*bar_width/4, -8, name2, ha='center', fontsize=9, fontweight='bold')
        
        x_pos += group_spacing
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'{title_prefix} Comparison: {name1} vs {name2}', fontsize=14, fontweight='bold')
    ax.set_ylim(-10, 100)
    ax.set_xlim(-0.5, x_pos - 0.3)
    ax.set_xticks(question_positions)
    ax.set_xticklabels(question_labels, fontsize=10, fontweight='bold', rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Create legend for responses
    handles = [mpatches.Patch(facecolor=response_colors[resp], edgecolor='white', label=resp) 
               for resp in all_responses]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
             ncol=4, fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    filename = f'viz/{filename_prefix}combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def compare_distributions(df1, df2, columns, name1, name2, title_prefix, filename_prefix):
    """Compare distributions side-by-side for specified columns."""
    for idx, col in enumerate(columns):
        if col not in df1.columns or col not in df2.columns:
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get value counts as percentages
        counts1 = df1[col].value_counts(normalize=True).sort_index() * 100
        counts2 = df2[col].value_counts(normalize=True).sort_index() * 100
        
        # Plot 1
        axes[0].barh(range(len(counts1)), counts1.values, color='steelblue')
        axes[0].set_yticks(range(len(counts1)))
        axes[0].set_yticklabels(counts1.index, fontsize=9)
        axes[0].set_xlabel('Percentage (%)')
        axes[0].set_title(f'{title_prefix}: {col.replace("_", " ").title()}\n{name1}', fontweight='bold')
        axes[0].set_xlim(0, max(counts1.values.max(), counts2.values.max()) * 1.1)
        
        # Add percentage labels
        for i, v in enumerate(counts1.values):
            axes[0].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        # Plot 2
        axes[1].barh(range(len(counts2)), counts2.values, color='coral')
        axes[1].set_yticks(range(len(counts2)))
        axes[1].set_yticklabels(counts2.index, fontsize=9)
        axes[1].set_xlabel('Percentage (%)')
        axes[1].set_title(f'{title_prefix}: {col.replace("_", " ").title()}\n{name2}', fontweight='bold')
        axes[1].set_xlim(0, max(counts1.values.max(), counts2.values.max()) * 1.1)
        
        # Add percentage labels
        for i, v in enumerate(counts2.values):
            axes[1].text(v + 1, i, f'{v:.1f}%', va='center', fontsize=9)
        
        plt.tight_layout()
        sanitized_col_name = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        filename = f'viz/{filename_prefix}{idx+1:02d}_{sanitized_col_name}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

def main():
    # Get file inputs
    file1 = input("Enter path to first survey CSV: ").strip()
    file2 = input("Enter path to second survey CSV: ").strip()
    name1 = input("Enter name for first survey (e.g., 'Human'): ").strip()
    name2 = input("Enter name for second survey (e.g., 'GPT'): ").strip()
    
    # Load data
    print(f"\nLoading {file1}...")
    df1 = load_and_prepare_data(file1)
    print(f"Loaded {len(df1)} rows")
    
    print(f"Loading {file2}...")
    df2 = load_and_prepare_data(file2)
    print(f"Loaded {len(df2)} rows")
    
    # Compare missingness
    compare_missingness(df1, df2, name1, name2)
    
    # Compare demographics distributions
    print(f"\n--- Plotting Demographics Comparison ---")
    compare_distributions(df1, df2, demo_cols, name1, name2, "Demographic", "compare_demo_")
    
    # Compare question distributions in a combined grid
    print(f"\n--- Plotting Question Responses Comparison (Combined) ---")
    compare_distributions_grid(df1, df2, question_cols, name1, name2, "Response", "compare_response_")
    
    print(f"\nComparison complete! Files saved to viz/ folder.")

if __name__ == "__main__":
    main()
