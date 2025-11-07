#!/usr/bin/env python3
"""
Visualization 1: Distribution of switch point positions (with transparent background)
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("../../data/spanglish_dataset.csv")

# Filter code-switched utterances
cs_utterances = df[df['has_code_switch'] == 'yes'].copy()

def find_first_switch_point(pos_str):
    """Find the index where the first language switch occurs"""
    tags = pos_str.split('|')
    for i in range(len(tags) - 1):
        if tags[i] != tags[i + 1]:
            return i + 1
    return len(tags)

# Compute first switch positions
cs_utterances['switch_point'] = cs_utterances['pos_tags'].apply(find_first_switch_point)

# --- 🟣 Print relevant information ---
num_cs = len(cs_utterances)
mean_switch = cs_utterances['switch_point'].mean()
median_switch = cs_utterances['switch_point'].median()
std_switch = cs_utterances['switch_point'].std()
min_switch = cs_utterances['switch_point'].min()
max_switch = cs_utterances['switch_point'].max()

print("Code-Switched Utterance Statistics:")
print(f"  Total CS utterances: {num_cs:,}")
print(f"  Mean first switch position: {mean_switch:.2f}")
print(f"  Median first switch position: {median_switch}")
print(f"  Standard deviation: {std_switch:.2f}")
print(f"  Min switch position: {min_switch}")
print(f"  Max switch position: {max_switch}")

# Optional summary message
if mean_switch < 5:
    print("\nMost switches occur early in the utterance.")
elif mean_switch < 10:
    print("\nSwitches typically occur near the middle of the utterance.")
else:
    print("\nSwitches tend to occur later in the utterance.")

# --- 🟣 Visualization ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('none')  # Transparent figure background
ax.patch.set_facecolor('none')   # Transparent axes background
ax.tick_params(colors='white')

# Plot histogram
ax.hist(cs_utterances['switch_point'], bins=20, color='#DC92FF', edgecolor='white', alpha=0.7)
ax.set_xlabel("Token Position of First Switch", fontsize=12, color='white')
ax.set_ylabel("Frequency", fontsize=12, color="white")
ax.set_title("Distribution of First Switch-Point Positions", fontsize=14, color='white')

# Subtle grid and white spines
ax.grid(axis='y', alpha=0.2, color='white')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for spine in ax.spines.values():
    spine.set_color('white')

# Save with transparent background
plt.tight_layout()
plt.savefig('../../figures/switch_point_distribution.png', 
            dpi=300, 
            bbox_inches='tight', 
            transparent=True)
plt.show()
