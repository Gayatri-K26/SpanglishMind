#!/usr/bin/env python3
"""
Hybrid Code-Switching Dataset Generator with Continuous Saving
Saves after each successful conversation to prevent data loss
"""

import requests
import json
import time
import random
import re
import os
from datetime import datetime
import spacy
from langdetect import detect_langs, LangDetectException
import nltk
from typing import List, Tuple, Dict

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Configuration
GOOGLE_API_KEY = "AIzaSyC2-uoVV_F_mMVzsc9WEu0F1Y4GAffoKzw"
GEMINI_MODEL = "gemini-2.0-flash"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GOOGLE_API_KEY}"

# File paths
OUTPUT_DIR = "../data"
JSON_FILE = os.path.join(OUTPUT_DIR, "spanglish_dataset.json")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "generation_checkpoint.json")

# Load spaCy models
try:
    nlp_en = spacy.load("en_core_web_sm")
    nlp_es = spacy.load("es_core_news_sm")
    print("✅ SpaCy models loaded")
except:
    print("⚠️ SpaCy models not found, using fallback")
    nlp_en = None
    nlp_es = None

class HybridLanguageTagger:
    """Combines HTML pre-tagging with professional NLP detection."""
    
    def __init__(self):
        self.nlp_en = nlp_en
        self.nlp_es = nlp_es
        
        self.spanish_markers = {
            'el', 'la', 'los', 'las', 'un', 'una', 'de', 'que', 'y', 'en',
            'por', 'para', 'con', 'sin', 'pero', 'cuando', 'donde', 'como',
            'está', 'están', 'estoy', 'mi', 'tu', 'su', 'qué', 'así', 'muy'
        }
        
        self.ambiguous = {'me', 'a', 'no', 'si', 'come', 'sale', 'dice'}
    
    def extract_and_tag(self, text: str, force_language: str = None) -> Tuple[List[Tuple[str, str]], List[str], str]:
        """
        Extract tokens from text with <ES> tags and detect languages.
        NEW: force_language parameter for monolingual conversations
        """
        
        # Pre-cleaning
        text = re.sub(r'\[\d+\]', '', text)
        
        tagged_tokens = []
        
        # NEW: Handle monolingual conversations
        if force_language in ['es', 'en']:
            words = text.split()
            for word in words:
                match = re.match(r'^(\W*)(.*?)(\W*)$', word)
                if match:
                    pre_punct, core_word, post_punct = match.groups()
                    
                    if pre_punct:
                        for p in pre_punct:
                            if p in '.,!?;:¿¡':
                                tagged_tokens.append((p, 'punct'))
                    
                    if core_word:
                        tagged_tokens.append((core_word, force_language))
                    
                    if post_punct:
                        for p in post_punct:
                            if p in '.,!?;:¿¡':
                                tagged_tokens.append((p, 'punct'))
        else:
            # Original logic for mixed conversations
            spanish_segments = []
            
            for match in re.finditer(r'<ES>(.*?)</ES>', text):
                spanish_content = match.group(1).strip()
                spanish_segments.append(spanish_content)
            
            text_with_placeholders = text
            for i, segment in enumerate(spanish_segments):
                text_with_placeholders = text_with_placeholders.replace(
                    f'<ES>{segment}</ES>', 
                    f' __SPANISH_{i}__ ', 
                    1
                )
            
            words = text_with_placeholders.split()
            
            for word in words:
                if word.startswith('__SPANISH_') and word.endswith('__'):
                    try:
                        idx = int(word.replace('__SPANISH_', '').replace('__', ''))
                        spanish_text = spanish_segments[idx]
                        for spanish_word in spanish_text.split():
                            if spanish_word:
                                tagged_tokens.append((spanish_word, 'es'))
                    except:
                        continue
                else:
                    match = re.match(r'^(\W*)(.*?)(\W*)$', word)
                    if match:
                        pre_punct, core_word, post_punct = match.groups()
                        
                        if pre_punct:
                            for p in pre_punct:
                                if p in '.,!?;:¿¡':
                                    tagged_tokens.append((p, 'punct'))
                        
                        if core_word:
                            if any(c in core_word for c in 'ñáéíóúü'):
                                tagged_tokens.append((core_word, 'es'))
                            elif core_word.lower() in self.spanish_markers:
                                tagged_tokens.append((core_word, 'es'))
                            else:
                                tagged_tokens.append((core_word, 'en'))
                        
                        if post_punct:
                            for p in post_punct:
                                if p in '.,!?;:¿¡':
                                    tagged_tokens.append((p, 'punct'))
        
        # Get POS tags
        pos_tags = self._get_pos_tags_simple(tagged_tokens)
        
        # Create clean text
        clean_text = re.sub(r'<ES>(.*?)</ES>', r'\1', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return tagged_tokens, pos_tags, clean_text
    
    def _get_pos_tags_simple(self, tagged_tokens: List[Tuple[str, str]]) -> List[str]:
        """Simple POS tagging fallback."""
        pos_tags = []
        for token, lang in tagged_tokens:
            if lang == 'punct':
                pos_tags.append(token)
            elif token.lower() in ['the', 'a', 'an', 'el', 'la', 'los', 'las']:
                pos_tags.append('DT')
            elif token.lower() in ['is', 'are', 'was', 'were', 'es', 'está', 'son']:
                pos_tags.append('VB')
            else:
                pos_tags.append('NN')
        return pos_tags

def load_checkpoint():
    """Load existing dataset or create new one."""
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✓ Loaded existing dataset: {len(data['conversations'])} conversations")
                return data['conversations'], data['metadata']
        except:
            pass
    
    print("Starting new dataset...")
    return [], {
        'generation_method': 'hybrid_tagged',
        'generation_date': datetime.now().isoformat(),
        'total_conversations': 0,
        'total_utterances': 0,
        'total_tokens': 0
    }

def save_checkpoint(conversations: List[Dict], metadata: Dict):
    """Save current progress to file."""
    metadata['total_conversations'] = len(conversations)
    metadata['total_utterances'] = sum(len(c['utterances']) for c in conversations)
    metadata['total_tokens'] = sum(sum(u['tokens'] for u in c['utterances']) for c in conversations)
    metadata['last_updated'] = datetime.now().isoformat()
    
    output = {
        'metadata': metadata,
        'conversations': conversations
    }
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    return metadata

def parse_conversation_flexible(text: str, tagger: HybridLanguageTagger, strategy_name: str = None, debug=False) -> List[Dict]:
    """
    Flexible parser that handles various speaker formats.
    NEW: strategy_name parameter to handle monolingual correctly
    """
    
    # Clean markdown
    text = text.replace('**Speaker A:**', 'Speaker A:')
    text = text.replace('**Speaker B:**', 'Speaker B:')
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Pre-clean
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\{\d+\}', '', text)
    
    utterances = []
    lines = text.split('\n')
    speaker_map = {}
    
    # NEW: Determine force_language based on strategy
    force_language = None
    if strategy_name == 'monolingual_spanish':
        force_language = 'es'
    elif strategy_name == 'monolingual_english':
        force_language = 'en'
    
    if debug and len(lines) > 0:
        print(f"    DEBUG: Found {len(lines)} lines to parse")
        print(f"    First line: {lines[0][:100] if lines else 'empty'}")
    
    for line in lines:
        line = line.strip()
        
        if not line or len(line) < 5:
            continue
        if line.lower().startswith(('example', 'format', 'instructions', 'note')):
            continue
        
        general_pattern = r'^([^:]+):\s*(.+)$'
        match = re.match(general_pattern, line)
        
        if match:
            speaker_raw = match.group(1).strip()
            content = match.group(2).strip()
            
            if content in ['...', '[text]', '[response]']:
                continue
            if len(content) < 3:
                continue
            
            # Map speaker names to Speaker A/B
            if speaker_raw not in speaker_map:
                if len(speaker_map) == 0:
                    speaker_map[speaker_raw] = "Speaker A"
                elif len(speaker_map) == 1:
                    speaker_map[speaker_raw] = "Speaker B"
                else:
                    speaker_map[speaker_raw] = "Speaker A" if len(utterances) % 2 == 0 else "Speaker B"
            
            speaker = speaker_map[speaker_raw]
            
            try:
                # NEW: Pass force_language to tagger
                tagged_tokens, pos_tags, clean_text = tagger.extract_and_tag(content, force_language)
                
                all_tokens = [t for t, _ in tagged_tokens if t]
                
                if all_tokens and len(all_tokens) > 2:
                    # Calculate actual language distribution
                    lang_counts = {'es': 0, 'en': 0}
                    for _, lang in tagged_tokens:
                        if lang in lang_counts:
                            lang_counts[lang] += 1
                    
                    total = lang_counts['es'] + lang_counts['en']
                    
                    # Correct CS detection based on actual languages
                    if lang_counts['es'] > 0 and lang_counts['en'] > 0:
                        has_cs = 'yes'
                        spanish_pct = (lang_counts['es'] / total * 100) if total > 0 else 0
                    else:
                        has_cs = 'no'
                        spanish_pct = 100.0 if lang_counts['es'] > 0 else 0.0
                    
                    utterances.append({
                        'speaker': speaker,
                        'text': clean_text,
                        'original_text': content,
                        'tokens': len([t for t, l in tagged_tokens if l != 'punct']),
                        'tagged_tokens': tagged_tokens,
                        'tokens_list': all_tokens,
                        'pos_tags': pos_tags,
                        'has_code_switch': has_cs,
                        'spanish_percentage': spanish_pct
                    })
                    
                    if debug:
                        print(f"    ✓ Parsed: {speaker_raw} → {speaker}")
            except Exception as e:
                if debug:
                    print(f"    Parse error: {str(e)[:100]}")
                continue
    
    return utterances

def generate_with_gemini(prompt: str, temperature: float = 0.8) -> str:
    """Call Gemini API with the prompt."""
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2000,
        }
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if result.get("candidates"):
                return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"    API Error: {e}")
    
    return None

# [Include all your get_strategy_prompt functions here - they stay the same]

def calculate_overall_cs_rate(conversations):
    """Calculate overall code-switching rate across all conversations."""
    total_utterances = 0
    total_cs_utterances = 0
    
    for conv in conversations:
        for utt in conv.get('utterances', []):
            total_utterances += 1
            if utt.get('has_code_switch') == 'yes':
                total_cs_utterances += 1
    
    cs_rate = (total_cs_utterances / total_utterances * 100) if total_utterances > 0 else 0
    return cs_rate, total_cs_utterances, total_utterances

def generate_dataset_continuous(target_conversations: int = 400):
    """Generate dataset with continuous saving."""
    print("="*70)
    print("HYBRID DATASET GENERATOR - CONTINUOUS SAVE MODE")
    print("="*70)
    
    # Load existing data
    conversations, metadata = load_checkpoint()
    existing_count = len(conversations)
    
    if existing_count >= target_conversations:
        print(f"✓ Already have {existing_count} conversations (target: {target_conversations})")
        return
    
    print(f"Target: {target_conversations} conversations")
    print(f"Starting from: {existing_count} existing conversations")
    print(f"Need to generate: {target_conversations - existing_count} more")
    print("="*70)
    
    # Initialize tagger
    tagger = HybridLanguageTagger()
    
    # Topics and parameters
    topics = [
        "weekend plans", "family dinner", "work meeting", "shopping trip",
        "movie night", "cooking together", "travel planning", "school project",
        "sports event", "birthday party", "restaurant visit", "health concerns"
    ]
    
    personas = [
        ("Mexican-American student", "Anglo roommate"),
        ("Dominican grandmother", "American granddaughter"),
        ("Colombian engineer", "Indian colleague"),
        ("Puerto Rican teacher", "Mixed-background parent")
    ]
    
    tones = ["casual", "formal", "emotional", "humorous"]
    
    # Continue from where we left off
    consecutive_failures = 0
    
    for i in range(existing_count, target_conversations):
        topic = random.choice(topics)
        
        # MODIFIED STRATEGY SELECTION FOR BALANCE
        # First 50 were mostly code-switching, now add monolingual
        # Adjust strategy selection for remaining conversations
        topic = random.choice(topics)
        
        # ULTRA-AGGRESSIVE MONOLINGUAL STRATEGY
        if i < 50:
            # Your original CS-heavy conversations
            strategy = (i % 5) + 1
        elif i >= 132:  # Everything after your current point
            # 95% monolingual, 5% minimal switch
            if random.random() < 0.95:
                strategy = 6 if random.random() < 0.5 else 7  # English or Spanish monolingual
            else:
                strategy = 8 
        else:  # Final 20
            # Back to mixed
            strategy = random.choice(range(1, 9))
        
        # Prepare strategy-specific parameters
        kwargs = {}
        if strategy == 1:
            kwargs['switch_points'] = sorted(random.sample(range(10, 50), 3))
            strategy_name = "controlled_switch"
        elif strategy == 2:
            kwargs['personaA'], kwargs['personaB'] = random.choice(personas)
            strategy_name = "role_based"
        elif strategy == 3:
            kwargs['tone'] = random.choice(tones)
            strategy_name = f"topic_variation_{kwargs['tone']}"
        elif strategy == 4:
            strategy_name = "back_translation"
        elif strategy == 5:
            strategy_name = "free_generation"
        elif strategy == 6:  # English only
            strategy_name = "monolingual_english"
            prompt = f"""Generate an English conversation about {topic}.
Use ONLY English. No Spanish at all.

Speaker A: [English only]
Speaker B: [English only]

Write 10 lines:"""
            
        elif strategy == 7:  # Spanish only
            strategy_name = "monolingual_spanish"
            prompt = f"""Genera una conversación en español sobre {topic}.
Solo español. Sin inglés.

Speaker A: [Solo español]
Speaker B: [Solo español]

10 líneas:"""
        else:  # strategy == 8
            strategy_name = "minimal_switch"
        
        print(f"\n[{i+1}/{target_conversations}] Strategy: {strategy_name} | Topic: {topic}")
        
        # Generate with retry
        success = False
        for attempt in range(3):
            # Get appropriate prompt based on strategy
            if strategy <= 5:
                prompt = get_strategy_prompt(strategy, topic, **kwargs)
            elif strategy == 6:  # Monolingual English
                prompt = f"""Generate a conversation about {topic} in ENGLISH ONLY.

IMPORTANT: This is a monolingual English conversation. NO Spanish words at all.
Write natural, fluent English dialogue.

Format:
Speaker A: [English only]
Speaker B: [English only]

Write 10-12 lines about {topic}:"""
            elif strategy == 7:  # Monolingual Spanish
                prompt = f"""Genera una conversación sobre {topic} SOLAMENTE EN ESPAÑOL.

IMPORTANTE: Conversación monolingüe en español. NO palabras en inglés.
Escribe diálogo natural y fluido en español.

Formato:
Speaker A: [Solo español]
Speaker B: [Solo español]

Escribe 10-12 líneas sobre {topic}:"""
            else:  # strategy == 8 - Minimal switch
                prompt = f"""Generate a mostly English conversation about {topic}.

Include only 1-2 SMALL Spanish words/phrases in the ENTIRE conversation.
Tag the few Spanish words with <ES></ES>.

Example (notice very minimal Spanish):
Speaker A: How was your day?
Speaker B: Pretty good, just got back from the store.
Speaker A: What did you buy?
Speaker B: Just some groceries for dinner.
Speaker A: Sounds good, <ES>gracias</ES> for picking that up.
Speaker B: No problem!

Generate 10-12 lines:"""
            
            response = generate_with_gemini(prompt, 0.7 + (attempt * 0.05))
            
            if response:
                # Use debug mode on failed attempts
                debug_mode = (attempt > 0)
                utterances = parse_conversation_flexible(response, tagger, debug=debug_mode)
                
                # If parsing failed completely, show what we got
                if not utterances and attempt == 0:
                    print(f"    ⚠️ Parse failed. Response preview:")
                    print(f"    {response[:200].replace(chr(10), ' | ')}")
                
                if utterances and len(utterances) >= 8:
                    # Success! Continue as before...
                    conversations.append({
                        'conversation_id': i,
                        'strategy': strategy_name,
                        'topic': topic,
                        'utterances': utterances
                    })
                    
                    # Calculate stats
                    total_tokens = sum(u['tokens'] for u in utterances)
                    cs_utterances = sum(1 for u in utterances if u['has_code_switch'] == 'yes')
                    
                    # Handle Spanish percentage for monolingual convos
                    if strategy == 6:  # English only
                        avg_spanish = 0.0
                    elif strategy == 7:  # Spanish only
                        avg_spanish = 100.0
                    else:
                        avg_spanish = sum(u['spanish_percentage'] for u in utterances) / len(utterances)
                    
                    print(f"  ✓ {len(utterances)} utterances | {total_tokens} tokens")
                    print(f"  ✓ {cs_utterances}/{len(utterances)} with CS | {avg_spanish:.1f}% Spanish")
                    
                    # Save checkpoint
                    metadata = save_checkpoint(conversations, metadata)
                    print(f"  💾 Saved! Total: {len(conversations)} conversations, {metadata['total_tokens']:,} tokens")
                    overall_cs_rate, total_cs, total_utt = calculate_overall_cs_rate(conversations)
                    
                    print(f"  📊 Overall CS Rate: {overall_cs_rate:.1f}% ({total_cs}/{total_utt} utterances)")
                    
                    success = True
                    consecutive_failures = 0
                    break

                else:
                    if utterances:
                        print(f"  ⚠️ Only {len(utterances)} utterances (need 8+), attempt {attempt+1}/3")
                    else:
                        print(f"  ⚠️ No utterances parsed, attempt {attempt+1}/3")
            else:
                print(f"  ❌ No API response, attempt {attempt+1}/3")
            
            if strategy in [6, 7, 8]:  # Monolingual or minimal
                time.sleep(2)  # Shorter delay for simpler prompts
            else:
                time.sleep(5)
        
        if not success:
            consecutive_failures += 1
            print(f"  ❌ Failed after 3 attempts")
            time.sleep(10) 
            if consecutive_failures < 5:
                print(f"     Skipping conversation {i+1}")
            else:
                print("\n⚠️ Too many consecutive failures. Stopping.")
                print(f"Progress saved: {len(conversations)} conversations")
                return
        
        # Rate limiting
        time.sleep(3)
    
    # Final summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total conversations: {len(conversations)}")
    print(f"Total utterances: {metadata['total_utterances']}")
    print(f"Total tokens: {metadata['total_tokens']:,}")
    print(f"Output: {JSON_FILE}")
    print("="*70)

# At the bottom of generate_dataset.py, after the main generation code

def recalculate_cs_rate(json_file):
    """Recalculate CS rate, excluding monolingual conversations."""
    import json
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_utterances = 0
    true_cs_utterances = 0  # Only count actual mixed utterances
    monolingual_spanish = 0
    monolingual_english = 0
    
    for conv in data['conversations']:
        for utt in conv['utterances']:
            total_utterances += 1
            tagged = utt.get('tagged_tokens', [])
            langs = set(l for t, l in tagged if l not in ['punct'])
            
            if 'en' in langs and 'es' in langs:
                true_cs_utterances += 1
            elif 'es' in langs and 'en' not in langs:
                monolingual_spanish += 1
            elif 'en' in langs and 'es' not in langs:
                monolingual_english += 1
    
    true_cs_rate = (true_cs_utterances / total_utterances * 100) if total_utterances > 0 else 0
    
    print("\n" + "="*70)
    print("TRUE CODE-SWITCHING ANALYSIS")
    print("="*70)
    print(f"Total utterances: {total_utterances}")
    print(f"Mixed bilingual utterances: {true_cs_utterances} ({true_cs_utterances/total_utterances*100:.1f}%)")
    print(f"Monolingual Spanish: {monolingual_spanish} ({monolingual_spanish/total_utterances*100:.1f}%)")
    print(f"Monolingual English: {monolingual_english} ({monolingual_english/total_utterances*100:.1f}%)")
    print(f"\nTrue CS Rate (bilingual mixed only): {true_cs_rate:.1f}%")
    print("="*70)

# Update your main execution
if __name__ == "__main__":
    # Generate dataset
    #generate_dataset_continuous(400)
    
    # After generation completes, analyze the true CS rate
    print("\nAnalyzing final dataset...")
    recalculate_cs_rate(JSON_FILE)