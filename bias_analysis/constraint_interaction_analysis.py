import json
import re
from collections import defaultdict

# Load your dataset
JSON_FILE = "../data/spanglish_dataset.json"

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

conversations = data['conversations']
print(f"Loaded {len(conversations)} conversations\n")

# ============================================================
# 1. MASKING EFFECT: FUNCTIONAL SWITCH SKEW ANALYSIS
# ============================================================
print("="*70)
print("1. FUNCTIONAL SWITCH SKEW ANALYSIS (Masking Effect)")
print("="*70)

# POS tag mappings to broader categories
POS_CATEGORIES = {
    'NOUN': ['NN', 'NNS', 'NNP', 'NNPS'],
    'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
    'ADJ': ['JJ', 'JJR', 'JJS'],
    'ADV': ['RB', 'RBR', 'RBS'],
    'DET': ['DT', 'PDT', 'WDT'],
    'PRON': ['PRP', 'PRP$', 'WP', 'WP$'],
    'PREP': ['IN', 'TO'],
    'CONJ': ['CC', 'IN'],
    'INTJ': ['UH'],
}

def get_broad_pos(tag):
    """Map specific POS tag to broad category."""
    for category, tags in POS_CATEGORIES.items():
        if tag in tags:
            return category
    return 'OTHER'

# Track POS by language
pos_by_language = {'es': defaultdict(int), 'en': defaultdict(int)}
total_by_pos = defaultdict(int)

# Track switches at word boundaries
switch_contexts = defaultdict(int)  # What POS triggers a switch?

# Analyze each utterance
for conv in conversations:
    for utt in conv['utterances']:
        tagged_tokens = utt.get('tagged_tokens', [])
        pos_tags = utt.get('pos_tags', [])
        
        # Ensure we have matching lengths
        if len(tagged_tokens) != len(pos_tags):
            continue
        
        prev_lang = None
        for i, ((token, lang), pos) in enumerate(zip(tagged_tokens, pos_tags)):
            if lang in ['es', 'en']:
                broad_pos = get_broad_pos(pos)
                pos_by_language[lang][broad_pos] += 1
                total_by_pos[broad_pos] += 1
                
                # Track switch points
                if prev_lang and prev_lang != lang and prev_lang in ['es', 'en']:
                    switch_contexts[f"{prev_lang}→{lang}:{broad_pos}"] += 1
                
                prev_lang = lang

# Calculate Functional Switch Skew
print("\n--- POS Distribution by Language ---")
print(f"\n{'POS Category':<12} | {'Spanish':>10} | {'English':>10} | {'ES %':>8} | {'EN %':>8} | {'Skew':>8}")
print("-"*70)

skew_data = []
for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'DET', 'INTJ', 'OTHER']:
    es_count = pos_by_language['es'][pos]
    en_count = pos_by_language['en'][pos]
    total = es_count + en_count
    
    if total > 0:
        es_pct = (es_count / total) * 100
        en_pct = (en_count / total) * 100
        # Skew: positive = English-dominant, negative = Spanish-dominant
        skew = en_pct - es_pct
        skew_data.append((pos, es_count, en_count, es_pct, en_pct, skew))
        print(f"{pos:<12} | {es_count:>10} | {en_count:>10} | {es_pct:>7.1f}% | {en_pct:>7.1f}% | {skew:>+7.1f}")

# Analyze what triggers switches
print("\n--- Switch Point Analysis ---")
print("What POS category does the speaker switch INTO?\n")

switch_into_es = defaultdict(int)
switch_into_en = defaultdict(int)

for context, count in switch_contexts.items():
    direction, pos = context.split(':')
    if direction == 'en→es':
        switch_into_es[pos] += count
    elif direction == 'es→en':
        switch_into_en[pos] += count

print(f"{'POS Category':<12} | {'→Spanish':>10} | {'→English':>10} | {'Direction Bias':>15}")
print("-"*55)

all_pos = set(switch_into_es.keys()) | set(switch_into_en.keys())
for pos in sorted(all_pos):
    to_es = switch_into_es[pos]
    to_en = switch_into_en[pos]
    total = to_es + to_en
    if total > 0:
        bias = "→English" if to_en > to_es else "→Spanish" if to_es > to_en else "Neutral"
        ratio = max(to_en, to_es) / min(to_en, to_es) if min(to_en, to_es) > 0 else float('inf')
        print(f"{pos:<12} | {to_es:>10} | {to_en:>10} | {bias:>10} ({ratio:.1f}x)")

# Calculate overall functional skew metric
print("\n--- Functional Switch Skew Metric ---")
noun_es_pct = pos_by_language['es']['NOUN'] / (pos_by_language['es']['NOUN'] + pos_by_language['en']['NOUN']) * 100 if (pos_by_language['es']['NOUN'] + pos_by_language['en']['NOUN']) > 0 else 0
verb_es_pct = pos_by_language['es']['VERB'] / (pos_by_language['es']['VERB'] + pos_by_language['en']['VERB']) * 100 if (pos_by_language['es']['VERB'] + pos_by_language['en']['VERB']) > 0 else 0
intj_es_pct = pos_by_language['es']['INTJ'] / (pos_by_language['es']['INTJ'] + pos_by_language['en']['INTJ']) * 100 if (pos_by_language['es']['INTJ'] + pos_by_language['en']['INTJ']) > 0 else 0

print(f"Spanish share of NOUNs: {noun_es_pct:.1f}%")
print(f"Spanish share of VERBs: {verb_es_pct:.1f}%")
print(f"Spanish share of INTJs: {intj_es_pct:.1f}%")
print(f"\nFunctional Skew Index (NOUN - VERB difference): {noun_es_pct - verb_es_pct:.1f}%")
print("(Positive = Spanish favored for nouns over verbs; Negative = opposite)")

# ============================================================
# 2. AMPLIFICATION EFFECT: ROLE/LOCATION PAIRING ANALYSIS
# ============================================================
print("\n" + "="*70)
print("2. ROLE/LOCATION PAIRING ANALYSIS (Amplification Effect)")
print("="*70)

# Technical/Expert vocabulary
TECH_EXPERT_TERMS = {
    'recommend', 'suggest', 'advise', 'think', 'believe', 'should', 'could',
    'would', 'must', 'need', 'important', 'best', 'better', 'option',
    'solution', 'approach', 'strategy', 'plan', 'idea', 'consider',
    'actually', 'definitely', 'probably', 'certainly', 'obviously',
    'professional', 'expert', 'experience', 'knowledge', 'skill',
    'system', 'process', 'method', 'technique', 'analyze', 'evaluate'
}

# Analyze role-based conversations
role_conversations = [c for c in conversations if 'role' in c.get('strategy', '').lower()]
print(f"\nFound {len(role_conversations)} role-based conversations")

# Track expert language by speaker position (A vs B)
expert_terms_by_speaker = {'Speaker A': {'en': 0, 'es': 0}, 'Speaker B': {'en': 0, 'es': 0}}
total_tokens_by_speaker = {'Speaker A': {'en': 0, 'es': 0}, 'Speaker B': {'en': 0, 'es': 0}}

# Track who gives advice vs asks questions
advice_markers = {'should', 'recommend', 'suggest', 'advise', 'try', 'consider', 'best', 'better'}
question_markers = {'?', 'how', 'what', 'why', 'when', 'where', 'which', 'can you', 'could you'}

advice_by_speaker = {'Speaker A': 0, 'Speaker B': 0}
questions_by_speaker = {'Speaker A': 0, 'Speaker B': 0}

for conv in role_conversations:
    for utt in conv['utterances']:
        speaker = utt.get('speaker', 'Unknown')
        text_lower = utt.get('text', '').lower()
        tagged_tokens = utt.get('tagged_tokens', [])
        
        if speaker not in expert_terms_by_speaker:
            continue
        
        # Count expert terms by language
        for token, lang in tagged_tokens:
            if lang in ['en', 'es']:
                total_tokens_by_speaker[speaker][lang] += 1
                if token.lower().strip('.,!?') in TECH_EXPERT_TERMS:
                    expert_terms_by_speaker[speaker][lang] += 1
        
        # Track advice vs questions
        if any(marker in text_lower for marker in advice_markers):
            advice_by_speaker[speaker] += 1
        if '?' in text_lower or any(text_lower.startswith(q) for q in ['how', 'what', 'why', 'when', 'where', 'which']):
            questions_by_speaker[speaker] += 1

print("\n--- Expert/Authority Language by Speaker ---")
print(f"\n{'Speaker':<12} | {'EN Expert':>10} | {'ES Expert':>10} | {'EN Total':>10} | {'ES Total':>10}")
print("-"*60)

for speaker in ['Speaker A', 'Speaker B']:
    en_exp = expert_terms_by_speaker[speaker]['en']
    es_exp = expert_terms_by_speaker[speaker]['es']
    en_tot = total_tokens_by_speaker[speaker]['en']
    es_tot = total_tokens_by_speaker[speaker]['es']
    print(f"{speaker:<12} | {en_exp:>10} | {es_exp:>10} | {en_tot:>10} | {es_tot:>10}")

print("\n--- Advice-Giving vs Question-Asking Pattern ---")
print(f"\n{'Speaker':<12} | {'Advice Given':>12} | {'Questions Asked':>15} | {'Role Pattern':>15}")
print("-"*60)

for speaker in ['Speaker A', 'Speaker B']:
    advice = advice_by_speaker[speaker]
    questions = questions_by_speaker[speaker]
    pattern = "Expert/Advisor" if advice > questions else "Learner/Seeker" if questions > advice else "Balanced"
    print(f"{speaker:<12} | {advice:>12} | {questions:>15} | {pattern:>15}")

# Analyze language dominance in expert utterances
print("\n--- Language of Expert Advice ---")

expert_utterances_by_lang = {'english_dominant': 0, 'spanish_dominant': 0, 'mixed': 0}

for conv in role_conversations:
    for utt in conv['utterances']:
        text_lower = utt.get('text', '').lower()
        tagged_tokens = utt.get('tagged_tokens', [])
        
        # Check if this is an advice-giving utterance
        if any(marker in text_lower for marker in advice_markers):
            lang_counts = {'en': 0, 'es': 0}
            for token, lang in tagged_tokens:
                if lang in lang_counts:
                    lang_counts[lang] += 1
            
            total = lang_counts['en'] + lang_counts['es']
            if total > 0:
                en_pct = lang_counts['en'] / total
                if en_pct > 0.7:
                    expert_utterances_by_lang['english_dominant'] += 1
                elif en_pct < 0.3:
                    expert_utterances_by_lang['spanish_dominant'] += 1
                else:
                    expert_utterances_by_lang['mixed'] += 1

total_expert = sum(expert_utterances_by_lang.values())
print(f"\nAdvice-giving utterances by dominant language:")
print(f"  English-dominant (>70% EN): {expert_utterances_by_lang['english_dominant']} ({expert_utterances_by_lang['english_dominant']/total_expert*100:.1f}%)" if total_expert > 0 else "  No data")
print(f"  Spanish-dominant (>70% ES): {expert_utterances_by_lang['spanish_dominant']} ({expert_utterances_by_lang['spanish_dominant']/total_expert*100:.1f}%)" if total_expert > 0 else "  No data")
print(f"  Mixed: {expert_utterances_by_lang['mixed']} ({expert_utterances_by_lang['mixed']/total_expert*100:.1f}%)" if total_expert > 0 else "  No data")

# ============================================================
# 3. COMPARE ROLE-BASED VS OTHER STRATEGIES
# ============================================================
print("\n" + "="*70)
print("3. ROLE-BASED VS OTHER STRATEGIES COMPARISON")
print("="*70)

# Calculate English dominance by strategy
strategy_lang_distribution = defaultdict(lambda: {'en': 0, 'es': 0})

for conv in conversations:
    strategy = conv.get('strategy', 'unknown')
    for utt in conv['utterances']:
        for token, lang in utt.get('tagged_tokens', []):
            if lang in ['en', 'es']:
                strategy_lang_distribution[strategy][lang] += 1

print(f"\n{'Strategy':<25} | {'English %':>10} | {'Spanish %':>10} | {'EN:ES Ratio':>12}")
print("-"*65)

for strategy in sorted(strategy_lang_distribution.keys()):
    en = strategy_lang_distribution[strategy]['en']
    es = strategy_lang_distribution[strategy]['es']
    total = en + es
    if total > 0:
        en_pct = en / total * 100
        es_pct = es / total * 100
        ratio = en / es if es > 0 else float('inf')
        print(f"{strategy:<25} | {en_pct:>9.1f}% | {es_pct:>9.1f}% | {ratio:>11.2f}:1")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY METRICS FOR REPORT")
print("="*70)

print("\n** MASKING EFFECT (Functional Switch Skew) **")
print(f"   - Spanish share of NOUNs: {noun_es_pct:.1f}%")
print(f"   - Spanish share of VERBs: {verb_es_pct:.1f}%")
print(f"   - Functional Skew Index: {noun_es_pct - verb_es_pct:+.1f}%")

print("\n** AMPLIFICATION EFFECT (Role/Location Pairing) **")
if total_expert > 0:
    en_expert_pct = expert_utterances_by_lang['english_dominant'] / total_expert * 100
    print(f"   - Expert advice in English-dominant utterances: {en_expert_pct:.1f}%")
print(f"   - Speaker A advice instances: {advice_by_speaker['Speaker A']}")
print(f"   - Speaker B advice instances: {advice_by_speaker['Speaker B']}")