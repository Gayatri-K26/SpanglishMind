#!/usr/bin/env python3
"""
Code-Switching Dataset Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import json

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_dataset():
    """Load the JSON dataset and convert to DataFrame."""
    print("Loading dataset...")
    
    json_path = "spanglish_dataset.json"
    
    # Load JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame format
    rows = []
    for conv in data['conversations']:
        conv_id = conv.get('conversation_id', 0)
        strategy = conv.get('strategy', 'unknown')
        topic = conv.get('topic', 'general')
        
        for utt_idx, utt in enumerate(conv.get('utterances', [])):
            # Extract tokens and POS tags from tagged_tokens
            tagged_tokens = utt.get('tagged_tokens', [])
            tokens = [t for t, lang in tagged_tokens if lang != 'punct']
            pos_tags = [lang.upper() for t, lang in tagged_tokens if lang != 'punct']
            
            row = {
                'conversation_id': conv_id,
                'utterance_id': utt_idx,
                'speaker': utt.get('speaker', 'Unknown'),
                'utterance': utt.get('text', ''),
                'tokens': '|'.join(tokens),
                'pos_tags': '|'.join(pos_tags),
                'has_code_switch': utt.get('has_code_switch', 'no'),
                'spanish_percentage': utt.get('spanish_percentage', 0.0),
                'generation_strategy': strategy,
                'topic': topic
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    print(f"✓ Loaded {len(df)} utterances from {len(data['conversations'])} conversations")
    return df

def prepare_analysis_data(df):
    """Prepare data for analysis."""
    # Parse tokens and POS tags
    df['token_list'] = df['tokens'].apply(lambda x: x.split('|') if pd.notna(x) else [])
    df['pos_list'] = df['pos_tags'].apply(lambda x: x.split('|') if pd.notna(x) else [])
    df['num_tokens'] = df['token_list'].apply(len)
    
    # Count code-switches per utterance
    def count_switches(pos_tags):
        if not pos_tags:
            return 0
        switches = 0
        for i in range(len(pos_tags) - 1):
            if pos_tags[i] != pos_tags[i+1] and pos_tags[i] in ['EN', 'ES'] and pos_tags[i+1] in ['EN', 'ES']:
                switches += 1
        return switches
    
    df['num_switches'] = df['pos_list'].apply(count_switches)
    
    # Calculate CSI (Code-Switching Index) - switches per token
    df['csi'] = df.apply(lambda row: row['num_switches'] / row['num_tokens'] 
                         if row['num_tokens'] > 0 else 0, axis=1)
    
    return df

# ============================================================================
# VISUALIZATION 1: Dataset Class Balance
# ============================================================================
def viz_class_balance(df):
    """Create bar chart showing distribution of language tokens."""
    print("\n" + "="*70)
    print("VISUALIZATION 1: Dataset Class Balance")
    print("="*70)
    
    # Count tokens by language
    token_counts = {'English': 0, 'Spanish': 0, 'Code-Switch': 0}
    
    for _, row in df.iterrows():
        pos_tags = row['pos_list']
        
        # Count consecutive sequences
        if not pos_tags:
            continue
            
        current_lang = pos_tags[0]
        is_switch_context = False
        
        for i, tag in enumerate(pos_tags):
            if tag == 'EN':
                # Check if this is near a switch
                if i > 0 and pos_tags[i-1] == 'ES':
                    is_switch_context = True
                elif i < len(pos_tags) - 1 and pos_tags[i+1] == 'ES':
                    is_switch_context = True
                else:
                    is_switch_context = False
                    
                if is_switch_context:
                    token_counts['Code-Switch'] += 1
                else:
                    token_counts['English'] += 1
                    
            elif tag == 'ES':
                # Check if this is near a switch
                if i > 0 and pos_tags[i-1] == 'EN':
                    is_switch_context = True
                elif i < len(pos_tags) - 1 and pos_tags[i+1] == 'EN':
                    is_switch_context = True
                else:
                    is_switch_context = False
                    
                if is_switch_context:
                    token_counts['Code-Switch'] += 1
                else:
                    token_counts['Spanish'] += 1
    
    total = sum(token_counts.values())
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = list(token_counts.keys())
    counts = list(token_counts.values())
    colors = ['#3498db', '#e74c3c', '#f39c12']
    
    bars = ax.bar(classes, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/total*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Token Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Token Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Language Classes in Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('viz1_class_balance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: viz1_class_balance.png")
    
    # Analysis
    print("\n ANALYSIS:")
    print(f"Total tokens: {total:,}")
    for cls, count in token_counts.items():
        print(f"  {cls}: {count:,} ({count/total*100:.1f}%)")
    
    # Calculate imbalance ratio
    max_class = max(token_counts.values())
    min_class = min(token_counts.values())
    imbalance_ratio = max_class / min_class
    
    print(f"\n Class Imbalance Analysis:")
    print(f"  Imbalance Ratio: {imbalance_ratio:.2f}:1")
    print(f"  Dominant Class: {max(token_counts, key=token_counts.get)}")
    print(f"  Minority Class: {min(token_counts, key=token_counts.get)}")
    
    print("\n Implications for Model Evaluation:")
    print("  • High class imbalance detected - accuracy alone is insufficient")
    print("  • Model could achieve high accuracy by simply predicting majority class")
    print("  • MUST use: Precision, Recall, F1-score for each class")
    print("  • Consider weighted F1 or macro-averaged metrics")
    print("  • Code-switch boundaries are critical but underrepresented")

# ============================================================================
# VISUALIZATION 2: Code-Switching Frequency Distribution
# ============================================================================
def viz_switch_frequency(df):
    """Create histogram of code-switch frequency per utterance."""
    print("\n" + "="*70)
    print("VISUALIZATION 2: Code-Switching Frequency Distribution")
    print("="*70)
    
    # Get switch counts
    switch_counts = df['num_switches'].values
    
    # Create bins: 0, 1-2, 3-5, 6-10, 11+
    def bin_switches(n):
        if n == 0:
            return '0'
        elif n <= 2:
            return '1-2'
        elif n <= 5:
            return '3-5'
        elif n <= 10:
            return '6-10'
        else:
            return '11+'
    
    df['switch_bin'] = df['num_switches'].apply(bin_switches)
    
    # Count utterances per bin
    bin_order = ['0', '1-2', '3-5', '6-10', '11+']
    bin_counts = df['switch_bin'].value_counts()
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    counts = [bin_counts.get(b, 0) for b in bin_order]
    colors_gradient = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(bin_order)))
    
    bars = ax.bar(bin_order, counts, color=colors_gradient, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Number of Code-Switches per Utterance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Utterances', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Code-Switching Frequency Across Dataset', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.yaxis.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('viz2_switch_frequency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: viz2_switch_frequency.png")
    
    # Analysis
    print("\n ANALYSIS:")
    print(f"Total utterances: {len(df):,}")
    for bin_name in bin_order:
        count = bin_counts.get(bin_name, 0)
        print(f"  {bin_name} switches: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Statistics
    mean_switches = df['num_switches'].mean()
    median_switches = df['num_switches'].median()
    max_switches = df['num_switches'].max()
    
    print(f"\n Statistical Summary:")
    print(f"  Mean switches per utterance: {mean_switches:.2f}")
    print(f"  Median switches per utterance: {median_switches:.1f}")
    print(f"  Maximum switches in single utterance: {int(max_switches)}")
    
    # Linguistic complexity assessment
    high_switch_pct = (df['num_switches'] >= 3).sum() / len(df) * 100
    no_switch_pct = (df['num_switches'] == 0).sum() / len(df) * 100
    
    print(f"\n Linguistic Complexity Characterization:")
    print(f"  Monolingual utterances: {no_switch_pct:.1f}%")
    print(f"  High-frequency switches (3+): {high_switch_pct:.1f}%")
    
    if no_switch_pct > 50:
        print("  → Dataset is SPARSELY code-switched (majority monolingual)")
    elif high_switch_pct > 30:
        print("  → Dataset is HIGHLY mixed bilingual")
    else:
        print("  → Dataset has MODERATE code-switching complexity")
    
    print("\n Predicted Model Failure Mode:")
    print("  • Model will likely have LOWEST RECALL on rare switch patterns")
    if high_switch_pct < 15:
        print("  • Specifically: Utterances with 6+ switches (data scarcity)")
    if no_switch_pct > 60:
        print("  • Model may over-predict monolingual sequences")
        print("  • Risk: Missing subtle code-switch boundaries in mixed contexts")

# ============================================================================
# VISUALIZATION 3: Language Biases in Syntactic Features
# ============================================================================
def viz_language_bias(df):
    """Compare POS tag frequencies between languages."""
    print("\n" + "="*70)
    print("VISUALIZATION 3: Language Biases in Syntactic Features")
    print("="*70)
    
    # Count top unigrams per language
    english_tokens = []
    spanish_tokens = []
    
    for _, row in df.iterrows():
        tokens = row['token_list']
        pos_tags = row['pos_list']
        
        for token, tag in zip(tokens, pos_tags):
            if tag == 'EN':
                english_tokens.append(token.lower())
            elif tag == 'ES':
                spanish_tokens.append(token.lower())
    
    # Get top 15 most frequent tokens for each language
    en_counter = Counter(english_tokens)
    es_counter = Counter(spanish_tokens)
    
    top_n = 15
    top_en = dict(en_counter.most_common(top_n))
    top_es = dict(es_counter.most_common(top_n))
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # English tokens
    en_words = list(top_en.keys())
    en_counts = list(top_en.values())
    ax1.barh(en_words, en_counts, color='#3498db', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Top 15 English Tokens', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Spanish tokens
    es_words = list(top_es.keys())
    es_counts = list(top_es.values())
    ax2.barh(es_words, es_counts, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Top 15 Spanish Tokens', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Language-Specific Token Frequency Comparison', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('viz3_language_bias.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: viz3_language_bias.png")
    
    # Analysis
    print("\n ANALYSIS:")
    print(f"Total English tokens: {len(english_tokens):,}")
    print(f"Total Spanish tokens: {len(spanish_tokens):,}")
    print(f"English/Spanish ratio: {len(english_tokens)/len(spanish_tokens):.2f}:1")
    
    # Lexical diversity
    en_unique = len(set(english_tokens))
    es_unique = len(set(spanish_tokens))
    en_diversity = en_unique / len(english_tokens) if english_tokens else 0
    es_diversity = es_unique / len(spanish_tokens) if spanish_tokens else 0
    
    print(f"\n Lexical Diversity:")
    print(f"  English unique tokens: {en_unique:,} (diversity: {en_diversity:.3f})")
    print(f"  Spanish unique tokens: {es_unique:,} (diversity: {es_diversity:.3f})")
    
    # Token distribution concentration
    en_top_pct = sum(top_en.values()) / len(english_tokens) * 100
    es_top_pct = sum(top_es.values()) / len(spanish_tokens) * 100
    
    print(f"\n Token Distribution Analysis:")
    print(f"  Top 15 English tokens cover: {en_top_pct:.1f}% of all English tokens")
    print(f"  Top 15 Spanish tokens cover: {es_top_pct:.1f}% of all Spanish tokens")
    
    if en_top_pct > es_top_pct + 5:
        print("  → English shows higher concentration (less syntactic variety)")
    elif es_top_pct > en_top_pct + 5:
        print("  → Spanish shows higher concentration (less syntactic variety)")
    else:
        print("  → Both languages show similar concentration patterns")
    
    print("\n Model Bias Implications:")
    if len(english_tokens) > len(spanish_tokens) * 1.5:
        print("  • English is DOMINANT in the dataset")
        print("  • Risk: Model will learn English grammar more effectively")
        print("  • Predicted bias: Lower accuracy on Spanish-dominant switches")
        print("  • Spanish code-switch boundaries may be under-detected")
    elif len(spanish_tokens) > len(english_tokens) * 1.5:
        print("  • Spanish is DOMINANT in the dataset")
        print("  • Risk: Model will learn Spanish grammar more effectively")
        print("  • Predicted bias: Lower accuracy on English-dominant switches")
        print("  • English code-switch boundaries may be under-detected")
    else:
        print("  • Languages are relatively BALANCED")
        print("  • Lower risk of systematic language bias")
        print("  • Model should generalize reasonably to both languages")

# ============================================================================
# VISUALIZATION 4: Sentence Length vs Code-Switching
# ============================================================================
def viz_length_vs_switching(df):
    """Visualize relationship between utterance length and code-switching."""
    print("\n" + "="*70)
    print("VISUALIZATION 4: Sentence Length vs Code-Switching")
    print("="*70)
    
    # Create length bins for box plot
    df['length_bin'] = pd.cut(df['num_tokens'], 
                              bins=[0, 5, 10, 15, 20, 100],
                              labels=['1-5', '6-10', '11-15', '16-20', '21+'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot
    ax1.scatter(df['num_tokens'], df['num_switches'], 
               alpha=0.4, s=30, c='#2ecc71', edgecolors='black', linewidth=0.5)
    
    # Add trend line
    z = np.polyfit(df['num_tokens'], df['num_switches'], 1)
    p = np.poly1d(z)
    ax1.plot(df['num_tokens'].sort_values(), 
            p(df['num_tokens'].sort_values()), 
            "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    
    ax1.set_xlabel('Utterance Length (tokens)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Code-Switches', fontsize=11, fontweight='bold')
    ax1.set_title('Scatter: Length vs Code-Switch Count', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Box plot
    df_plot = df[df['length_bin'].notna()]
    df_plot.boxplot(column='csi', by='length_bin', ax=ax2, 
                   patch_artist=True, grid=False)
    ax2.set_xlabel('Utterance Length (tokens)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Code-Switching Index (CSI)', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot: CSI Distribution by Length', fontsize=12, fontweight='bold')
    plt.suptitle('')  # Remove default title
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('viz4_length_vs_switching.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: viz4_length_vs_switching.png")
    
    # Statistical analysis
    correlation = df['num_tokens'].corr(df['num_switches'])
    correlation_csi = df['num_tokens'].corr(df['csi'])
    
    print("\n ANALYSIS:")
    print(f"Correlation (length vs switch count): {correlation:.3f}")
    print(f"Correlation (length vs CSI): {correlation_csi:.3f}")
    
    # Interpretation
    if abs(correlation) > 0.7:
        strength = "STRONG"
    elif abs(correlation) > 0.4:
        strength = "MODERATE"
    else:
        strength = "WEAK"
    
    direction = "positive" if correlation > 0 else "negative"
    
    print(f"\n🔍 Correlation Assessment:")
    print(f"  Strength: {strength} {direction} correlation")
    
    # Group analysis
    grouped = df.groupby('length_bin').agg({
        'num_switches': 'mean',
        'csi': 'mean',
        'num_tokens': 'count'
    }).round(2)
    
    print(f"\n Mean Statistics by Length Group:")
    print(grouped)
    
    print("\n Critical Assessment:")
    if abs(correlation) > 0.5:
        print("  • Correlation EXISTS between length and switching frequency")
        print("\n  Is this a LINGUISTIC INSIGHT or GENERATION ARTIFACT?")
        print("  → LIKELY ARTIFACT of LLM generation process because:")
        print("    1. Prompt may have encouraged longer = more complex = more switches")
        print("    2. LLM tends to add switches to demonstrate bilingualism")
        print("    3. Real-world CS often brief insertions (not length-dependent)")
        print("\n  Implications for Model Generalization:")
        print("    • Model may OVERFIT to this length-switching pattern")
        print("    • Risk: Poor performance on real data where CS is independent of length")
        print("    • May fail on: Short utterances with switches, long monolingual sequences")
    else:
        print("  • NO strong correlation between length and switching")
        print("  → This is MORE REALISTIC linguistically")
        print("  → Code-switching often driven by topic/pragmatics, not length")
        print("\n  Implications for Model Generalization:")
        print("    • Model less likely to use length as spurious feature")
        print("    • Better potential for real-world generalization")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all visualizations and analyses."""
    print("="*70)
    print("CODE-SWITCHING DATASET ANALYSIS")
    print("Comprehensive Linguistic Characterization")
    print("="*70)
    
    # Load and prepare data
    df = load_dataset()
    df = prepare_analysis_data(df)
    
    # Generate all visualizations
    viz_class_balance(df)
    viz_switch_frequency(df)
    viz_language_bias(df)
    viz_length_vs_switching(df)
    
    print("\n" + "="*70)
    print(" ALL VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. viz1_class_balance.png")
    print("  2. viz2_switch_frequency.png")
    print("  3. viz3_language_bias.png")
    print("  4. viz4_length_vs_switching.png")
    print("\nAll visualizations include detailed statistical analysis and")
    print("implications for model performance and evaluation metrics.")
    print("="*70)

if __name__ == "__main__":
    main()