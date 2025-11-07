#!/usr/bin/env python3
"""
Rate-limit resilient monolingual conversation generator
Features exponential backoff and resume capability
"""

import json
import time
import random
import os
from datetime import datetime
import requests

# Your existing configuration
GOOGLE_API_KEY = "AIzaSyC2-uoVV_F_mMVzsc9WEu0F1Y4GAffoKzw"
GEMINI_MODEL = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"

def calculate_needed_conversations(current_file: str, target_cs_rate: float = 0.25):
    """Calculate how many monolingual conversations needed to reach target CS rate."""
    
    with open(current_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Current statistics
    total_utterances = 0
    cs_utterances = 0
    
    for conv in data['conversations']:
        for utt in conv['utterances']:
            total_utterances += 1
            if utt.get('has_code_switch') == 'yes':
                cs_utterances += 1
    
    current_cs_rate = cs_utterances / total_utterances if total_utterances > 0 else 0
    
    print(f"Current stats:")
    print(f"  Total utterances: {total_utterances}")
    print(f"  CS utterances: {cs_utterances}")
    print(f"  Current CS rate: {current_cs_rate:.1%}")
    
    # Calculate how many monolingual utterances needed
    needed_total = cs_utterances / target_cs_rate
    needed_monolingual_utterances = needed_total - total_utterances
    
    # Assume ~10 utterances per conversation
    needed_conversations = max(0, int(needed_monolingual_utterances / 10) + 1)
    
    print(f"\nTo reach {target_cs_rate:.0%} CS rate:")
    print(f"  Need ~{needed_monolingual_utterances:.0f} more monolingual utterances")
    print(f"  Need ~{needed_conversations} more monolingual conversations")
    
    return needed_conversations, cs_utterances, total_utterances

def generate_batch_prompt(topics_batch, language='english'):
    """Generate multiple conversations in ONE API call to save on rate limits."""
    
    if language == 'english':
        prompt = f"""Generate {len(topics_batch)} separate English-only conversations on these topics:

{chr(10).join([f'{i+1}. {topic}' for i, topic in enumerate(topics_batch)])}

CRITICAL RULES:
- ENGLISH ONLY - absolutely no Spanish words
- Each conversation should be 8-10 lines
- Natural, casual dialogue
- Use format: "Speaker A:" and "Speaker B:"
- Clearly separate each conversation with "---CONVERSATION N---" (where N is the conversation number)

Now generate all {len(topics_batch)} conversations:"""
    
    else:  # Spanish
        prompt = f"""Genera {len(topics_batch)} conversaciones separadas SOLO EN ESPAÑOL sobre estos temas:

{chr(10).join([f'{i+1}. {topic}' for i, topic in enumerate(topics_batch)])}

REGLAS:
- SOLO ESPAÑOL - ninguna palabra en inglés
- Cada conversación: 8-10 líneas
- Formato: "Speaker A:" y "Speaker B:"
- Separa con "---CONVERSACIÓN N---" (donde N es el número)

Genera las {len(topics_batch)} conversaciones:"""
    
    return prompt

def parse_batch_response(response_text: str, language: str, topics_batch: list):
    """Parse multiple conversations from a single API response."""
    
    conversations = []
    
    # Split by conversation markers
    conv_patterns = ['---CONVERSATION', '---CONVERSACIÓN', '---Conversation', '---conversación']
    
    parts = response_text
    for pattern in conv_patterns:
        parts = parts.replace(pattern, '|||SPLIT|||')
    
    segments = parts.split('|||SPLIT|||')
    
    for i, segment in enumerate(segments[1:], 0):  # Skip first empty segment
        if i >= len(topics_batch):
            break
            
        utterances = []
        lines = segment.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
                
            # Parse speaker lines
            if 'Speaker A:' in line or 'Speaker B:' in line:
                if 'Speaker A:' in line:
                    speaker = 'Speaker A'
                    text = line.split('Speaker A:', 1)[1].strip()
                else:
                    speaker = 'Speaker B'
                    text = line.split('Speaker B:', 1)[1].strip()
                
                if text and len(text) > 2:
                    # Create monolingual utterance structure
                    tokens = text.split()
                    utterances.append({
                        'speaker': speaker,
                        'text': text,
                        'original_text': text,
                        'tokens': len(tokens),
                        'tokens_list': tokens,
                        'has_code_switch': 'no',
                        'spanish_percentage': 100.0 if language == 'spanish' else 0.0,
                        'tagged_tokens': [[t, 'es' if language == 'spanish' else 'en'] for t in tokens]
                    })
        
        if len(utterances) >= 6:  # Accept if we got at least 6 utterances
            conversations.append({
                'strategy': f'monolingual_{language}',
                'topic': topics_batch[i] if i < len(topics_batch) else 'general',
                'utterances': utterances
            })
    
    return conversations

def save_checkpoint(filename: str, generated: int, api_calls: int, last_batch: int):
    """Save checkpoint for resuming."""
    checkpoint = {
        'generated': generated,
        'api_calls': api_calls,
        'last_batch': last_batch,
        'timestamp': datetime.now().isoformat()
    }
    checkpoint_file = filename.replace('.json', '_checkpoint.json')
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def load_checkpoint(filename: str):
    """Load checkpoint if exists."""
    checkpoint_file = filename.replace('.json', '_checkpoint.json')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

def generate_monolingual_efficiently(output_file: str, needed_conversations: int):
    """Generate monolingual conversations efficiently with rate limit handling."""
    
    # Load existing data
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    existing_count = len(data['conversations'])
    
    # Check for checkpoint
    checkpoint = load_checkpoint(output_file)
    if checkpoint:
        print(f"\n📌 Found checkpoint from {checkpoint['timestamp']}")
        print(f"   Resuming from: {checkpoint['generated']} conversations generated")
        generated = checkpoint['generated']
        api_calls = checkpoint['api_calls']
        start_batch = checkpoint['last_batch'] + 1
    else:
        generated = 0
        api_calls = 0
        start_batch = 1
    
    # Topics pool
    topics = [
        "weekend plans", "cooking dinner", "office meeting", "gym workout",
        "movie review", "book club", "grocery shopping", "car repair",
        "vacation planning", "birthday party", "home renovation", "pet care",
        "weather chat", "sports game", "restaurant visit", "doctor appointment",
        "school project", "job interview", "morning routine", "coffee break",
        "hiking trip", "music concert", "art museum", "beach day",
        "game night", "study session", "cleaning day", "garden work",
        "family dinner", "online shopping", "tech support", "road trip",
        "gardening tips", "cooking show", "fitness goals", "meditation session"
    ]
    
    print(f"\nGenerating {needed_conversations - generated} more monolingual conversations...")
    print("="*70)
    
    consecutive_failures = 0
    wait_time = 15  # Start with 15 seconds
    
    batch_num = start_batch
    
    while generated < needed_conversations:
        # Adaptive batch size based on how many we still need
        remaining = needed_conversations - generated
        if remaining <= 3:
            batch_size = remaining
        elif consecutive_failures > 2:
            batch_size = 2  # Reduce batch size if hitting limits
        else:
            batch_size = min(5, remaining)
        
        # Randomly select topics
        batch_topics = random.sample(topics, batch_size)
        
        # Alternate between English and Spanish
        language = 'english' if batch_num % 2 == 1 else 'spanish'
        
        print(f"\nBatch {batch_num}: Generating {batch_size} {language} conversations...")
        
        # Generate batch prompt
        prompt = generate_batch_prompt(batch_topics, language)
        
        # Make API call with retry logic
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 3000,
            }
        }
        
        success = False
        for retry in range(3):  # Try up to 3 times per batch
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("candidates"):
                        response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Parse all conversations from response
                        new_conversations = parse_batch_response(response_text, language, batch_topics)
                        
                        if new_conversations:
                            # Add conversation IDs and append to dataset
                            for conv in new_conversations:
                                conv['conversation_id'] = existing_count + generated
                                data['conversations'].append(conv)
                                generated += 1
                                
                                print(f"  ✓ Added conversation {conv['conversation_id']}: {conv['topic']} ({len(conv['utterances'])} utterances)")
                            
                            api_calls += 1
                            
                            # Save progress
                            data['metadata']['total_conversations'] = len(data['conversations'])
                            data['metadata']['last_updated'] = datetime.now().isoformat()
                            
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                            
                            # Save checkpoint
                            save_checkpoint(output_file, generated, api_calls, batch_num)
                            
                            print(f"  💾 Saved progress: {generated}/{needed_conversations} conversations")
                            
                            # Calculate current CS rate
                            total_utt = 0
                            cs_utt = 0
                            for c in data['conversations']:
                                for u in c['utterances']:
                                    total_utt += 1
                                    if u.get('has_code_switch') == 'yes':
                                        cs_utt += 1
                            
                            current_rate = (cs_utt / total_utt * 100) if total_utt > 0 else 0
                            print(f"  📊 Current CS rate: {current_rate:.1f}%")
                            
                            success = True
                            consecutive_failures = 0
                            wait_time = max(15, wait_time - 5)  # Reduce wait time on success
                            break
                        else:
                            print(f"  ⚠️ No valid conversations parsed")
                    else:
                        print(f"  ⚠️ No response from API")
                        
                elif response.status_code == 429:
                    # Rate limit hit - exponential backoff
                    consecutive_failures += 1
                    wait_time = min(300, wait_time * 2)  # Double wait time, max 5 minutes
                    print(f"  ⚠️ Rate limit hit (429). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    print(f"  ❌ API error: {response.status_code}")
                    
            except Exception as e:
                print(f"  ❌ Error: {str(e)[:100]}")
            
            if not success and retry < 2:
                print(f"  Retry {retry + 2}/3 in {wait_time} seconds...")
                time.sleep(wait_time)
        
        if success:
            batch_num += 1
            # Standard wait between successful batches
            if generated < needed_conversations:
                print(f"  Waiting {wait_time} seconds before next batch...")
                time.sleep(wait_time)
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("\n⚠️ Too many failures. Stopping generation.")
                print(f"Generated {generated} conversations so far. You can resume later.")
                break
            else:
                # Longer wait after failure
                wait_time = min(300, wait_time * 1.5)
                print(f"  Failed batch. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
    
    # Clean up checkpoint if complete
    if generated >= needed_conversations:
        checkpoint_file = output_file.replace('.json', '_checkpoint.json')
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print("\n✅ Generation complete! Checkpoint file removed.")
    
    print("\n" + "="*70)
    print(f"Generated {generated} conversations with {api_calls} API calls")
    if api_calls > 0:
        print(f"Efficiency: {generated/api_calls:.1f} conversations per API call")
    print("="*70)

if __name__ == "__main__":
    # File paths
    input_file = "data/hybrid_dataset_fixed_cleaned_retagged.json"  # Use your fixed dataset
    
    # Calculate how many monolingual conversations needed
    needed, cs_utts, total_utts = calculate_needed_conversations(input_file, target_cs_rate=0.25)
    
    if needed > 0:
        print("\nProceed with generation? (y/n): ", end="")
        if input().lower() == 'y':
            generate_monolingual_efficiently(input_file, needed)
            
            # Final statistics
            print("\nRecalculating final statistics...")
            calculate_needed_conversations(input_file, target_cs_rate=0.25)
        else:
            print("Generation cancelled.")
    else:
        print("\n✅ Already at or below target CS rate!")