#!/usr/bin/env python3
"""
Visualization 2: Token-level language distribution pie chart (white text)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("../../data/spanglish_dataset.csv")

def count_language_tokens(pos_str):
    """Count EN and ES tokens from POS tags"""
    tags = str(pos_str).split('|')
    en_count = sum(1 for t in tags if t == 'EN')
    es_count = sum(1 for t in tags if t == 'ES')
    return pd.Series({"English": en_count, "Spanish": es_count})

# Count tokens by language
lang_counts = df['pos_tags'].apply(count_language_tokens)
lang_totals = lang_counts.sum()

# Print statistics
print("Token counts by language:")
print(f"  English: {lang_totals['English']:,}")
print(f"  Spanish: {lang_totals['Spanish']:,}")
print(f"  Total: {lang_totals.sum():,}")
print(f"  English %: {lang_totals['English']/lang_totals.sum()*100:.1f}%")
print(f"  Spanish %: {lang_totals['Spanish']/lang_totals.sum()*100:.1f}%")

# Create pie chart
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('none')
ax.patch.set_facecolor('none')
colors = ["#DC92FF", "#2C643C"]

# Pie chart with white text
wedges, texts, autotexts = plt.pie(
    lang_totals.values,
    labels=lang_totals.index,
    autopct='%1.1f%%',
    colors=colors,
    explode=(0.05, 0.05),
    shadow=True,
    startangle=90,
    textprops={'color': 'white', 'fontsize': 12}
)

# Ensure label texts are white
for text in texts:
    text.set_color('white')
for autotext in autotexts:
    autotext.set_color('white')

# Title (white)
plt.title("Token-Level Language Distribution", fontsize=14, fontweight='bold', color='white')

plt.tight_layout()
plt.savefig('../../figures/language_distribution.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
