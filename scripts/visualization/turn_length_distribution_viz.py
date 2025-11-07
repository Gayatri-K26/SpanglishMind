#!/usr/bin/env python3
"""
Visualization 3: Utterance length distribution by code-switching status (white text)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Load CSV
df = pd.read_csv("../../data/spanglish_dataset.csv")

# Compute number of tokens per utterance
df['token_count'] = df['tokens'].apply(lambda x: len(str(x).split('|')))

# Separate data by code-switching status
cs_yes = df[df['has_code_switch'] == 'yes']['token_count']
cs_no = df[df['has_code_switch'] == 'no']['token_count']

# Create figure with transparent background
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')

# Create box plot
bp = ax.boxplot(
    [cs_no, cs_yes],
    labels=['No Code-Switch', 'Code-Switch'],
    patch_artist=True,
    medianprops=dict(color='white', linewidth=2),
    boxprops=dict(facecolor='#92FFB0', alpha=0.7),
    whiskerprops=dict(color='white', linewidth=1.5),
    capprops=dict(color='white', linewidth=1.5),
    flierprops=dict(marker='o', markerfacecolor='white', markeredgecolor='white', markersize=5, alpha=0.8)
)

# Color code-switch box differently
bp['boxes'][1].set_facecolor('#DC92FF')

# Labels and title (white)
ax.set_ylabel("Number of Tokens", fontsize=12, color='white')
ax.set_xlabel("Code-Switching Status", fontsize=12, color='white')
ax.set_title("Utterance Length Distribution by Code-Switching", fontsize=14, fontweight='bold', color='white')

# White ticks and spines
ax.tick_params(axis='both', colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

# Subtle white grid
ax.grid(axis='y', alpha=0.2, color='white')

# Save with transparent background
plt.tight_layout()
plt.savefig('../../figures/utterance_lengths.png',
            dpi=300,
            bbox_inches='tight',
            transparent=True)
plt.show()

# Print statistics
print("Utterance length statistics:")
print(f"\nNon-CS utterances (n={len(cs_no)}):")
print(f"  Mean: {cs_no.mean():.1f} tokens")
print(f"  Median: {cs_no.median():.1f} tokens")
print(f"  Std: {cs_no.std():.1f} tokens")
print(f"  Min/Max: {cs_no.min()}/{cs_no.max()} tokens")

print(f"\nCS utterances (n={len(cs_yes)}):")
print(f"  Mean: {cs_yes.mean():.1f} tokens")
print(f"  Median: {cs_yes.median():.1f} tokens")
print(f"  Std: {cs_yes.std():.1f} tokens")
print(f"  Min/Max: {cs_yes.min()}/{cs_yes.max()} tokens")

# Statistical test
statistic, pvalue = stats.mannwhitneyu(cs_no, cs_yes)
print(f"\nMann-Whitney U test:")
print(f"  Statistic: {statistic}")
print(f"  P-value: {pvalue:.4f}")
if pvalue < 0.05:
    print("  Result: Significant difference in lengths")
else:
    print("  Result: No significant difference in lengths")
