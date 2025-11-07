#!/usr/bin/env python3
"""
Retag dataset using the <ES> tags from original_text field
"""

import json
import re
import os

def extract_spanish_segments(original_text):
    """Extract Spanish segments marked with <ES> tags."""
    spanish_segments = []
    
    # Find all Spanish tagged content
    for match in re.finditer(r'<ES>(.*?)</ES>', original_text):
        spanish_content = match.group(1).strip()
        # Split into words
        words = spanish_content.split()
        spanish_segments.extend(words)
    
    # Clean punctuation attached to words
    cleaned = []
    for word in spanish_segments:
        # Preserve the word with its punctuation but mark it as Spanish
        cleaned.append(word)
    
    return set(cleaned)  # Return as set for fast lookup

def retag_utterance(utterance):
    """Retag an utterance based on original_text ES tags."""
    
    original_text = utterance.get('original_text', '')
    
    # If no original_text or no ES tags, skip
    if not original_text:
        return False
    
    # Extract Spanish words from ES tags
    spanish_words = extract_spanish_segments(original_text)
    
    # If no Spanish marked in original, skip
    if not spanish_words:
        return False
    
    # Retag the tagged_tokens
    new_tagged_tokens = []
    modified = False
    
    for token, current_lang in utterance.get('tagged_tokens', []):
        if current_lang == 'punct':
            # Keep punctuation as is
            new_tagged_tokens.append([token, 'punct'])
        else:
            # Check if this token should be Spanish
            # Remove punctuation for comparison
            clean_token = token.strip('.,!?¿¡;:')
            
            # Check if token is in Spanish words (case-insensitive)
            is_spanish = False
            for spanish_word in spanish_words:
                clean_spanish = spanish_word.strip('.,!?¿¡;:')
                if clean_token.lower() == clean_spanish.lower():
                    is_spanish = True
                    break
                # Also check if token contains the Spanish word
                if clean_token in spanish_word or spanish_word in clean_token:
                    is_spanish = True
                    break
            
            if is_spanish and current_lang != 'es':
                new_tagged_tokens.append([token, 'es'])
                modified = True
            elif not is_spanish and current_lang != 'en':
                new_tagged_tokens.append([token, 'en'])
                modified = True
            else:
                new_tagged_tokens.append([token, current_lang])
    
    if modified:
        utterance['tagged_tokens'] = new_tagged_tokens
        
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
    
    return modified

def retag_dataset(input_file, output_file=None):
    """Retag entire dataset using original_text ES tags."""
    
    if output_file is None:
        output_file = input_file.replace('.json', '_retagged.json')
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_utterances = 0
    modified_utterances = 0
    
    # Process each conversation
    for conv_idx, conv in enumerate(data['conversations']):
        for utt_idx, utterance in enumerate(conv['utterances']):
            total_utterances += 1
            
            # Check if this has original_text with ES tags
            original = utterance.get('original_text', '')
            if '<ES>' in original:
                if retag_utterance(utterance):
                    modified_utterances += 1
                    
                    # Show some examples
                    if modified_utterances <= 5:
                        print(f"\nExample {modified_utterances}:")
                        print(f"  Original: {original[:80]}...")
                        print(f"  Spanish %: {utterance['spanish_percentage']:.1f}%")
                        print(f"  CS Status: {utterance['has_code_switch']}")
    
    # Also fix the text field (remove ES tags and artifacts)
    for conv in data['conversations']:
        for utterance in conv['utterances']:
            text = utterance.get('text', '')
            # Remove ES tags
            text = re.sub(r'<ES>', '', text)
            text = re.sub(r'</ES>', '', text)
            # Fix artifacts
            text = text.replace('bolsa<', 'bolsa')
            text = text.replace('huevos<', 'huevos') 
            text = text.replace('Vamos<', 'Vamos')
            text = text.replace('<.', '.')
            text = text.replace('< ', ' ')
            text = re.sub(r'\s+', ' ', text).strip()
            utterance['text'] = text
    
    # Save retagged dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Retagging complete!")
    print(f"Modified {modified_utterances} out of {total_utterances} utterances")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    # Calculate final statistics
    total_utt = 0
    cs_utt = 0
    spanish_only = 0
    english_only = 0
    
    for conv in data['conversations']:
        for utt in conv['utterances']:
            total_utt += 1
            if utt.get('has_code_switch') == 'yes':
                cs_utt += 1
            elif utt.get('spanish_percentage', 0) == 100:
                spanish_only += 1
            elif utt.get('spanish_percentage', 0) == 0:
                english_only += 1
    
    cs_rate = (cs_utt / total_utt * 100) if total_utt > 0 else 0
    
    print(f"\nFinal statistics:")
    print(f"Total utterances: {total_utt}")
    print(f"Code-switched utterances: {cs_utt} ({cs_utt/total_utt*100:.1f}%)")
    print(f"Monolingual Spanish: {spanish_only} ({spanish_only/total_utt*100:.1f}%)")
    print(f"Monolingual English: {english_only} ({english_only/total_utt*100:.1f}%)")
    print(f"Overall CS rate: {cs_rate:.1f}%")
    
    return output_file

if __name__ == "__main__":
    import sys
    
    # Default to the fixed dataset
    input_file = "data/hybrid_dataset_fixed_cleaned.json"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)
    
    retagged_file = retag_dataset(input_file)
    
    print(f"\nYou can now use the retagged dataset:")
    print(f"python3 mono_generation.py {retagged_file}")