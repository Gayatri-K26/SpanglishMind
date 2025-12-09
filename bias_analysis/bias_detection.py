import json
import re
from collections import defaultdict
import pandas as pd

# Load your dataset
JSON_FILE = "../data/spanglish_dataset.json"

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

conversations = data['conversations']
print(f"Loaded {len(conversations)} conversations\n")

# ============================================================
# 1. ROLE/EXPERTISE BIAS ANALYSIS
# ============================================================
print("="*60)
print("1. ROLE/EXPERTISE BIAS ANALYSIS")
print("="*60)

TECH_TERMS = {'bug','deploy','api','architecture','database','model',
              'epoch','latency','optimize','merge','pull','request',
              'code','server','function','algorithm','debug','system'}

tech_by_language = {'en': 0, 'es': 0}
tech_by_strategy = defaultdict(int)
total_tokens_by_lang = {'en': 0, 'es': 0}

for conv in conversations:
    strategy = conv.get('strategy', 'unknown')
    for utt in conv['utterances']:
        text_lower = utt['text'].lower()
        tech_count = sum(1 for term in TECH_TERMS if term in text_lower)
        
        # Count by dominant language in utterance
        tagged = utt.get('tagged_tokens', [])
        lang_counts = {'en': 0, 'es': 0}
        for token, lang in tagged:
            if lang in lang_counts:
                lang_counts[lang] += 1
        
        dominant_lang = 'en' if lang_counts['en'] >= lang_counts['es'] else 'es'
        tech_by_language[dominant_lang] += tech_count
        tech_by_strategy[strategy] += tech_count
        total_tokens_by_lang['en'] += lang_counts['en']
        total_tokens_by_lang['es'] += lang_counts['es']

print("\nTechnical Terms by Dominant Language:")
print(f"  English-dominant utterances: {tech_by_language['en']} technical terms")
print(f"  Spanish-dominant utterances: {tech_by_language['es']} technical terms")
print(f"  Ratio (EN:ES): {tech_by_language['en']/(tech_by_language['es']+0.001):.2f}:1")

print(f"\nTotal Tokens by Language:")
print(f"  English: {total_tokens_by_lang['en']:,}")
print(f"  Spanish: {total_tokens_by_lang['es']:,}")

# Normalized rate (tech terms per 1000 tokens)
en_rate = (tech_by_language['en'] / total_tokens_by_lang['en']) * 1000 if total_tokens_by_lang['en'] > 0 else 0
es_rate = (tech_by_language['es'] / total_tokens_by_lang['es']) * 1000 if total_tokens_by_lang['es'] > 0 else 0
print(f"\nTech Terms per 1000 Tokens:")
print(f"  English: {en_rate:.2f}")
print(f"  Spanish: {es_rate:.2f}")

# ============================================================
# 2. LOCATION/CULTURAL BIAS ANALYSIS
# ============================================================
print("\n" + "="*60)
print("2. LOCATION/CULTURAL BIAS ANALYSIS")
print("="*60)

CULTURAL_TERMS = {
    'hispanic_stereotypes': {'fiesta','siesta','taco','burrito','salsa',
                            'abuela','familia','mijo','mija','compadre'},
    'general_cultural': {'festival','traditional','cultural','heritage',
                        'celebration','customs','homeland'}
}

cultural_counts = {'en_dominant': defaultdict(int), 'es_dominant': defaultdict(int)}
stereotype_by_lang = {'en': 0, 'es': 0}

for conv in conversations:
    for utt in conv['utterances']:
        text_lower = utt['text'].lower()
        tagged = utt.get('tagged_tokens', [])
        lang_counts = {'en': 0, 'es': 0}
        for token, lang in tagged:
            if lang in lang_counts:
                lang_counts[lang] += 1
        
        dominant = 'en_dominant' if lang_counts['en'] >= lang_counts['es'] else 'es_dominant'
        dominant_simple = 'en' if lang_counts['en'] >= lang_counts['es'] else 'es'
        
        for category, terms in CULTURAL_TERMS.items():
            for term in terms:
                if term in text_lower:
                    cultural_counts[dominant][category] += 1
        
        for term in CULTURAL_TERMS['hispanic_stereotypes']:
            if term in text_lower:
                stereotype_by_lang[dominant_simple] += 1

print("\nCultural/Stereotype Terms by Language Dominance:")
for lang_type, counts in cultural_counts.items():
    print(f"\n  {lang_type}:")
    for category, count in counts.items():
        print(f"    {category}: {count}")

print(f"\nHispanic Stereotype Terms Distribution:")
print(f"  In English-dominant utterances: {stereotype_by_lang['en']}")
print(f"  In Spanish-dominant utterances: {stereotype_by_lang['es']}")

# Analyze persona patterns
print("\nPersona Analysis from Strategy Names:")
persona_strategies = [c['strategy'] for c in conversations if 'role' in c.get('strategy','').lower()]
print(f"  Role-based conversations: {len(persona_strategies)}")

# ============================================================
# 3. TOPIC/DOMAIN BIAS ANALYSIS
# ============================================================
print("\n" + "="*60)
print("3. TOPIC/DOMAIN BIAS ANALYSIS")
print("="*60)

DOMAIN_CATEGORIES = {
    'domestic': {'dinner','cooking','family','home','kitchen','food','eat','meal'},
    'social': {'party','birthday','weekend','movie','friend','fun','shopping'},
    'professional': {'work','meeting','project','deadline','boss','office','job'},
    'technical': {'code','system','data','analysis','research','study'}
}

topic_by_strategy = defaultdict(lambda: defaultdict(int))
topic_by_spanish_pct = {'high_spanish': defaultdict(int), 'low_spanish': defaultdict(int)}

for conv in conversations:
    strategy = conv.get('strategy', 'unknown')
    topic = conv.get('topic', 'unknown')
    
    # Calculate avg Spanish percentage for conversation
    spanish_pcts = [u.get('spanish_percentage', 0) for u in conv['utterances']]
    avg_spanish = sum(spanish_pcts) / len(spanish_pcts) if spanish_pcts else 0
    spanish_level = 'high_spanish' if avg_spanish > 50 else 'low_spanish'
    
    # Categorize topic
    topic_lower = topic.lower()
    for domain, keywords in DOMAIN_CATEGORIES.items():
        if any(kw in topic_lower for kw in keywords):
            topic_by_strategy[strategy][domain] += 1
            topic_by_spanish_pct[spanish_level][domain] += 1
            break
    else:
        topic_by_strategy[strategy]['other'] += 1
        topic_by_spanish_pct[spanish_level]['other'] += 1

print("\nTopic Distribution by Spanish Percentage:")
print("\n  High Spanish (>50%):")
for domain, count in sorted(topic_by_spanish_pct['high_spanish'].items()):
    print(f"    {domain}: {count}")

print("\n  Low Spanish (<=50%):")
for domain, count in sorted(topic_by_spanish_pct['low_spanish'].items()):
    print(f"    {domain}: {count}")

# Topic frequency overall
print("\nOverall Topic Distribution:")
topic_counts = defaultdict(int)
for conv in conversations:
    topic_counts[conv.get('topic', 'unknown')] += 1

for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
    print(f"  {topic}: {count}")

# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

total_utterances = sum(len(c['utterances']) for c in conversations)
cs_utterances = sum(1 for c in conversations for u in c['utterances'] if u.get('has_code_switch')=='yes')

print(f"\nDataset Overview:")
print(f"  Total conversations: {len(conversations)}")
print(f"  Total utterances: {total_utterances}")
print(f"  Code-switching utterances: {cs_utterances} ({cs_utterances/total_utterances*100:.1f}%)")
print(f"  English tokens: {total_tokens_by_lang['en']:,}")
print(f"  Spanish tokens: {total_tokens_by_lang['es']:,}")
print(f"  Language ratio (EN:ES): {total_tokens_by_lang['en']/(total_tokens_by_lang['es']+1):.2f}:1")