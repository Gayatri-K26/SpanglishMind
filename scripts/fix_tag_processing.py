#!/usr/bin/env python3
"""
Fix HTML tag processing issues in the dataset
"""

import json
import re
import os

def fix_tag_issues(text):
    """Fix various tag processing issues in text."""
    # First, remove any <ES> </ES> tags completely (they should already be processed)
    text = re.sub(r'<ES>', '', text)
    text = re.sub(r'</ES>', '', text)
    
    # Fix specific known issues
    text = text.replace('bolsa<.', 'bolsa.')
    text = text.replace('huevos<', 'huevos')
    text = text.replace('Vamos<.', 'Vamos.')
    
    # Remove any standalone < or >
    text = text.replace('<.', '.')
    text = text.replace('< ', ' ')
    text = text.replace(' >', ' ')
    text = text.replace('<', '')
    text = text.replace('>', '')
    
    # Fix merged ES tags in tokens
    text = text.replace('ES¿', '¿')
    text = text.replace('ES¡', '¡')
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def fix_dataset(input_file, output_file=None):
    """Fix tag processing issues in entire dataset."""
    
    if output_file is None:
        output_file = input_file.replace('.json', '_cleaned.json')
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    fixed_count = 0
    utterance_count = 0
    
    # Process each conversation
    for conv in data['conversations']:
        for utterance in conv['utterances']:
            utterance_count += 1
            
            # Fix the text field
            original_text = utterance.get('text', '')
            fixed_text = fix_tag_issues(original_text)
            
            if original_text != fixed_text:
                fixed_count += 1
                print(f"  Fixed: '{original_text[:50]}...' -> '{fixed_text[:50]}...'")
                utterance['text'] = fixed_text
            
            # Fix tokens_list
            new_tokens_list = []
            for token in utterance.get('tokens_list', []):
                # Remove ES prefix from tokens
                if token.startswith('ES'):
                    token = token[2:]
                # Remove trailing < or >
                token = token.rstrip('<>').lstrip('<>')
                if token:
                    new_tokens_list.append(token)
            
            utterance['tokens_list'] = new_tokens_list
            
            # Fix tagged_tokens
            new_tagged_tokens = []
            for token_pair in utterance.get('tagged_tokens', []):
                if len(token_pair) == 2:
                    token, lang = token_pair
                    # Clean the token
                    if token.startswith('ES'):
                        token = token[2:]
                    token = token.rstrip('<>').lstrip('<>')
                    
                    if token:
                        new_tagged_tokens.append([token, lang])
            
            utterance['tagged_tokens'] = new_tagged_tokens
            
            # Recalculate token count
            utterance['tokens'] = len([t for t, l in new_tagged_tokens if l != 'punct'])
            
            # Recalculate Spanish percentage
            lang_counts = {'es': 0, 'en': 0}
            for token, lang in new_tagged_tokens:
                if lang in lang_counts:
                    lang_counts[lang] += 1
            
            total = lang_counts['es'] + lang_counts['en']
            if total > 0:
                utterance['spanish_percentage'] = (lang_counts['es'] / total) * 100
                
                # Update code-switching status
                if lang_counts['es'] > 0 and lang_counts['en'] > 0:
                    utterance['has_code_switch'] = 'yes'
                else:
                    utterance['has_code_switch'] = 'no'
    
    # Save cleaned dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Cleaning complete!")
    print(f"Fixed {fixed_count} utterances out of {utterance_count}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    # Calculate final statistics
    total_utt = 0
    cs_utt = 0
    for conv in data['conversations']:
        for utt in conv['utterances']:
            total_utt += 1
            if utt.get('has_code_switch') == 'yes':
                cs_utt += 1
    
    cs_rate = (cs_utt / total_utt * 100) if total_utt > 0 else 0
    print(f"\nFinal statistics:")
    print(f"Total utterances: {total_utt}")
    print(f"Code-switched utterances: {cs_utt}")
    print(f"Code-switching rate: {cs_rate:.1f}%")
    
    return output_file

if __name__ == "__main__":
    import sys
    
    # Default to the fixed dataset
    input_file = "data/hybrid_dataset_fixed.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)
    
    cleaned_file = fix_dataset(input_file)
    
    print(f"\nNow you can continue generating monolingual conversations using:")
    print(f"python3 mono_generation.py {cleaned_file}")