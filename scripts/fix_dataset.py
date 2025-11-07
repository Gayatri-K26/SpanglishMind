#!/usr/bin/env python3
"""
Fix existing dataset by using strategy labels to correct tagging
"""

import json
import os
from typing import List, Tuple, Dict

def fix_monolingual_conversations(json_file: str, output_file: str = None):
    """
    Fix tagging in dataset based on conversation strategy.
    
    Args:
        json_file: Path to the existing dataset JSON
        output_file: Path for fixed output (if None, adds '_fixed' to original name)
    """
    
    # Load the dataset
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations_fixed = 0
    utterances_fixed = 0
    
    # Process each conversation
    for conv in data['conversations']:
        strategy = conv.get('strategy', '')
        
        # Fix monolingual Spanish conversations
        if strategy == 'monolingual_spanish':
            conversations_fixed += 1
            
            for utterance in conv['utterances']:
                utterances_fixed += 1
                
                # Get the existing tagged tokens
                tagged_tokens = utterance.get('tagged_tokens', [])
                
                # Re-tag all non-punctuation tokens as Spanish
                new_tagged_tokens = []
                for token, lang in tagged_tokens:
                    if lang == 'punct':
                        new_tagged_tokens.append([token, 'punct'])
                    else:
                        # Change everything else to Spanish
                        new_tagged_tokens.append([token, 'es'])
                
                # Update the utterance
                utterance['tagged_tokens'] = new_tagged_tokens
                utterance['has_code_switch'] = 'no'  # No code-switching in monolingual
                utterance['spanish_percentage'] = 100.0
        
        # Fix monolingual English conversations
        elif strategy == 'monolingual_english':
            conversations_fixed += 1
            
            for utterance in conv['utterances']:
                utterances_fixed += 1
                
                # Get the existing tagged tokens
                tagged_tokens = utterance.get('tagged_tokens', [])
                
                # Re-tag all non-punctuation tokens as English
                new_tagged_tokens = []
                for token, lang in tagged_tokens:
                    if lang == 'punct':
                        new_tagged_tokens.append([token, 'punct'])
                    else:
                        # Change everything else to English
                        new_tagged_tokens.append([token, 'en'])
                
                # Update the utterance
                utterance['tagged_tokens'] = new_tagged_tokens
                utterance['has_code_switch'] = 'no'  # No code-switching in monolingual
                utterance['spanish_percentage'] = 0.0
        
        # For minimal_switch strategy, ensure low Spanish percentage
        elif strategy == 'minimal_switch':
            for utterance in conv['utterances']:
                # Count languages in the utterance
                tagged_tokens = utterance.get('tagged_tokens', [])
                lang_counts = {'es': 0, 'en': 0}
                
                for token, lang in tagged_tokens:
                    if lang in lang_counts:
                        lang_counts[lang] += 1
                
                total = lang_counts['es'] + lang_counts['en']
                if total > 0:
                    spanish_pct = (lang_counts['es'] / total) * 100
                    utterance['spanish_percentage'] = spanish_pct
                    
                    # Check if there's actual code-switching
                    if lang_counts['es'] > 0 and lang_counts['en'] > 0:
                        utterance['has_code_switch'] = 'yes'
                    else:
                        utterance['has_code_switch'] = 'no'
    
    # Recalculate metadata statistics
    total_utterances = 0
    total_tokens = 0
    cs_utterances = 0
    monolingual_spanish_utts = 0
    monolingual_english_utts = 0
    mixed_utts = 0
    
    for conv in data['conversations']:
        for utt in conv['utterances']:
            total_utterances += 1
            total_tokens += utt.get('tokens', 0)
            
            if utt.get('has_code_switch') == 'yes':
                cs_utterances += 1
                mixed_utts += 1
            else:
                # Check if monolingual Spanish or English
                spanish_pct = utt.get('spanish_percentage', 0)
                if spanish_pct == 100.0:
                    monolingual_spanish_utts += 1
                elif spanish_pct == 0.0:
                    monolingual_english_utts += 1
    
    # Update metadata
    data['metadata']['total_utterances'] = total_utterances
    data['metadata']['total_tokens'] = total_tokens
    data['metadata']['code_switching_utterances'] = cs_utterances
    data['metadata']['monolingual_spanish_utterances'] = monolingual_spanish_utts
    data['metadata']['monolingual_english_utterances'] = monolingual_english_utts
    data['metadata']['mixed_bilingual_utterances'] = mixed_utts
    data['metadata']['fixes_applied'] = {
        'conversations_fixed': conversations_fixed,
        'utterances_fixed': utterances_fixed,
        'fix_method': 'strategy-based-retagging'
    }
    
    # Prepare output filename
    if output_file is None:
        base_name = os.path.splitext(json_file)[0]
        output_file = f"{base_name}_fixed.json"
    
    # Save the fixed dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("="*70)
    print("DATASET FIXING COMPLETE")
    print("="*70)
    print(f"Input file: {json_file}")
    print(f"Output file: {output_file}")
    print(f"Conversations fixed: {conversations_fixed}")
    print(f"Utterances fixed: {utterances_fixed}")
    print("-"*70)
    print("UPDATED STATISTICS:")
    print(f"Total utterances: {total_utterances}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Mixed bilingual utterances: {mixed_utts} ({mixed_utts/total_utterances*100:.1f}%)")
    print(f"Monolingual Spanish: {monolingual_spanish_utts} ({monolingual_spanish_utts/total_utterances*100:.1f}%)")
    print(f"Monolingual English: {monolingual_english_utts} ({monolingual_english_utts/total_utterances*100:.1f}%)")
    print(f"Code-switching rate: {cs_utterances/total_utterances*100:.1f}%")
    print("="*70)
    
    return data

def verify_fix(json_file: str):
    """
    Verify the fix by checking a few examples.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("\nVERIFYING FIX - Sample Check:")
    print("-"*70)
    
    # Find and display examples of each strategy type
    strategies_to_check = ['monolingual_spanish', 'monolingual_english', 'minimal_switch']
    
    for strategy in strategies_to_check:
        # Find first conversation with this strategy
        for conv in data['conversations']:
            if conv.get('strategy') == strategy:
                print(f"\nStrategy: {strategy}")
                print(f"Conversation ID: {conv['conversation_id']}")
                
                # Show first utterance
                if conv['utterances']:
                    utt = conv['utterances'][0]
                    print(f"Text: {utt['text'][:100]}...")
                    print(f"Has code-switch: {utt['has_code_switch']}")
                    print(f"Spanish %: {utt['spanish_percentage']:.1f}%")
                    
                    # Show first few tokens
                    tokens_sample = utt['tagged_tokens'][:10]
                    print(f"First 10 tokens: {tokens_sample}")
                break
    
    print("-"*70)

if __name__ == "__main__":
    import sys
    
    # Default file paths
    input_file = "data/hybrid_dataset.json"
    
    # Check if file path was provided as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
        print("Usage: python fix_dataset.py [path_to_json_file]")
        sys.exit(1)
    
    # Fix the dataset
    print(f"Fixing dataset: {input_file}")
    fixed_data = fix_monolingual_conversations(input_file)
    
    # Verify the fix
    output_file = os.path.splitext(input_file)[0] + "_fixed.json"
    verify_fix(output_file)