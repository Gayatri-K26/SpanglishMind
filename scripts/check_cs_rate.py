#!/usr/bin/env python3
"""
Simple dataset analysis script to verify rubric compliance.
"""

import json
import numpy as np
from collections import Counter

# File paths
JSON_FILE = "../data/spanglish_dataset.json"

def load_dataset():
    """Load the JSON dataset."""
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_code_switching(data):
    """Analyze code-switching patterns in the dataset."""
    conversations = data['conversations']
    
    stats = {
        'total_utterances': 0,
        'code_switched_utterances': 0,
        'pure_english': 0,
        'pure_spanish': 0,
        'tokens_by_lang': {'en': 0, 'es': 0},
        'switch_types': {'intersentential': 0, 'intrasentential': 0},
        'utterance_lengths': [],
        'cs_utterance_lengths': [],
        'spanish_percentages': []
    }
    
    for conv in conversations:
        for utt in conv['utterances']:
            stats['total_utterances'] += 1
            
            # Get language tags
            tagged_tokens = utt.get('tagged_tokens', [])
            languages = [lang for _, lang in tagged_tokens if lang in ['en', 'es']]
            
            if not languages:
                continue
            
            # Count tokens by language
            lang_counts = Counter(languages)
            stats['tokens_by_lang']['en'] += lang_counts.get('en', 0)
            stats['tokens_by_lang']['es'] += lang_counts.get('es', 0)
            
            # Track utterance length
            utt_length = len(languages)
            stats['utterance_lengths'].append(utt_length)
            
            # Determine if code-switched
            unique_langs = set(languages)
            is_code_switched = len(unique_langs) > 1
            
            if is_code_switched:
                stats['code_switched_utterances'] += 1
                stats['cs_utterance_lengths'].append(utt_length)
                
                # Calculate Spanish percentage for this utterance
                es_pct = (lang_counts.get('es', 0) / len(languages)) * 100
                stats['spanish_percentages'].append(es_pct)
                
                # Detect switch type (simplified)
                switches = sum(1 for i in range(len(languages)-1) 
                             if languages[i] != languages[i+1])
                if switches >= 2:
                    stats['switch_types']['intrasentential'] += 1
                else:
                    stats['switch_types']['intersentential'] += 1
            else:
                # Pure language utterances
                if 'en' in unique_langs:
                    stats['pure_english'] += 1
                else:
                    stats['pure_spanish'] += 1
    
    return stats

def print_statistics(stats, metadata):
    """Print detailed statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Token counts
    total_tokens = metadata.get('total_tokens', 0)
    print(f"\n📊 TOKEN COUNTS:")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  English tokens: {stats['tokens_by_lang']['en']:,}")
    print(f"  Spanish tokens: {stats['tokens_by_lang']['es']:,}")
    
    total_lang_tokens = stats['tokens_by_lang']['en'] + stats['tokens_by_lang']['es']
    if total_lang_tokens > 0:
        en_pct = (stats['tokens_by_lang']['en'] / total_lang_tokens) * 100
        es_pct = (stats['tokens_by_lang']['es'] / total_lang_tokens) * 100
        print(f"  Language distribution: {en_pct:.1f}% EN, {es_pct:.1f}% ES")
    
    # Utterance counts
    print(f"\n💬 UTTERANCE COUNTS:")
    print(f"  Total utterances: {stats['total_utterances']}")
    print(f"  Code-switched utterances: {stats['code_switched_utterances']}")
    print(f"  Pure English utterances: {stats['pure_english']}")
    print(f"  Pure Spanish utterances: {stats['pure_spanish']}")
    
    # Code-switching rate
    if stats['total_utterances'] > 0:
        cs_rate = (stats['code_switched_utterances'] / stats['total_utterances']) * 100
        print(f"  Code-switch rate: {cs_rate:.1f}%")
        
        print(f"\n🎯 RUBRIC COMPLIANCE:")
        if 20 <= cs_rate <= 30:
            print(f"  ✓ Code-switch rate {cs_rate:.1f}% is within 20-30% target")
        else:
            print(f"  ⚠️ Code-switch rate {cs_rate:.1f}% is outside 20-30% target")
        
        if total_tokens >= 15000:
            print(f"  ✓ Token count {total_tokens:,} exceeds 15,000 minimum")
        else:
            print(f"  ⚠️ Token count {total_tokens:,} is below 15,000 minimum")
    
    # Switch types
    print(f"\n🔀 CODE-SWITCH TYPES:")
    total_cs = stats['switch_types']['intersentential'] + stats['switch_types']['intrasentential']
    if total_cs > 0:
        inter_pct = (stats['switch_types']['intersentential'] / total_cs) * 100
        intra_pct = (stats['switch_types']['intrasentential'] / total_cs) * 100
        print(f"  Intersentential: {stats['switch_types']['intersentential']} ({inter_pct:.1f}%)")
        print(f"  Intrasentential: {stats['switch_types']['intrasentential']} ({intra_pct:.1f}%)")
    
    # Utterance length stats
    if stats['utterance_lengths']:
        print(f"\n📏 UTTERANCE LENGTH STATISTICS:")
        print(f"  Mean length: {np.mean(stats['utterance_lengths']):.1f} tokens")
        print(f"  Median length: {np.median(stats['utterance_lengths']):.1f} tokens")
        print(f"  Min/Max: {min(stats['utterance_lengths'])}/{max(stats['utterance_lengths'])} tokens")
    
    print("="*70)
    

def main():
    """Main analysis pipeline."""
    print("="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    
    # Load data
    data = load_dataset()
    metadata = data.get('metadata', {})
    
    # Analyze code-switching
    stats = analyze_code_switching(data)
    
    # Print statistics
    print_statistics(stats, metadata)
  
    print("\n✅ Analysis complete")
    

if __name__ == "__main__":
    main()