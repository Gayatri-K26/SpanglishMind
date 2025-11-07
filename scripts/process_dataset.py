#!/usr/bin/env python3
"""
Process your retagged dataset to final CSV format.
Works with hybrid_dataset_fixed_cleaned_retagged.json
"""

import json
import csv
import os

# File paths
INPUT_JSON = "data/spanglish_dataset.json"
OUTPUT_CSV = "data/spanglish_dataset.csv"

def process_to_csv():
    """Convert your retagged JSON dataset to CSV format."""
    
    print("="*70)
    print("DATASET PROCESSOR - JSON TO CSV")
    print("="*70)
    
    # Load JSON data
    print(f"\n📂 Loading dataset from: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    conversations = data.get("conversations", [])
    metadata = data.get("metadata", {})
    
    print(f"✓ Loaded {len(conversations)} conversations")
    
    # Process and write CSV
    print(f"\n📝 Converting to CSV format...")
    
    row_id = 0
    total_cs_utterances = 0
    total_utterances = 0
    strategy_counts = {}
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header (simplified for your needs)
        writer.writerow([
            "id",                    
            "conversation_id",       
            "utterance_id",         
            "speaker",              
            "utterance",            # Clean text
            "tokens",               # Pipe-separated tokens
            "pos_tags",             # Pipe-separated POS tags
            "has_code_switch",      # yes/no
            "spanish_percentage",   # Percentage of Spanish
            "generation_strategy",  
            "topic"                
        ])
        
        # Process each conversation
        for conv in conversations:
            conv_id = conv.get("conversation_id", 0)
            strategy = conv.get("strategy", "unknown")
            topic = conv.get("topic", "general")
            
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Process each utterance
            for utt_idx, utt in enumerate(conv.get("utterances", [])):
                total_utterances += 1
                
                # Extract data
                speaker = utt.get("speaker", "Unknown")
                clean_text = utt.get("text", "")
                
                # Extract tokens from tagged_tokens (excluding punctuation)
                tokens_list = []
                pos_tags_list = []
                
                for token, lang in utt.get("tagged_tokens", []):
                    if lang != "punct":  # Skip punctuation
                        tokens_list.append(token)
                        # Simple POS tagging based on language
                        if lang == "es":
                            pos_tags_list.append("ES")
                        elif lang == "en":
                            pos_tags_list.append("EN")
                        else:
                            pos_tags_list.append("UNK")
                
                # Join with pipe separator
                tokens_str = "|".join(tokens_list)
                pos_tags_str = "|".join(pos_tags_list)
                
                # Code-switch detection
                has_cs = utt.get("has_code_switch", "no")
                if has_cs == "yes":
                    total_cs_utterances += 1
                
                spanish_pct = utt.get("spanish_percentage", 0)
                
                # Write row
                writer.writerow([
                    row_id,                     
                    conv_id,                    
                    utt_idx,                    
                    speaker,                    
                    clean_text,                
                    tokens_str,                
                    pos_tags_str,              
                    has_cs,                    
                    f"{spanish_pct:.1f}",      
                    strategy,                  
                    topic                      
                ])
                
                row_id += 1
    
    # Calculate statistics
    cs_rate = (total_cs_utterances / total_utterances * 100) if total_utterances > 0 else 0
    
    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"✓ Total rows written: {row_id}")
    print(f"✓ Total utterances: {total_utterances}")
    print(f"✓ Code-switched utterances: {total_cs_utterances}")
    print(f"✓ Code-switch rate: {cs_rate:.1f}%")
    print(f"✓ Output saved to: {OUTPUT_CSV}")
    
    print("\n📊 STRATEGY DISTRIBUTION:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  - {strategy}: {count} conversations")

if __name__ == "__main__":
    process_to_csv()