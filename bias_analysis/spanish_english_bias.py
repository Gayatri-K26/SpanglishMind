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
# 1. AFFECTIVE SKEW ANALYSIS
# ============================================================
print("="*70)
print("1. AFFECTIVE SKEW ANALYSIS")
print("="*70)

# Emotional/affective lexicons by language
SPANISH_AFFECTIVE = {
    # Interjections & exclamations
    'ay', 'órale', 'híjole', 'ándale', 'guau', 'uy', 'epa', 'oye', 'mira',
    # Endearments & diminutives
    'mijo', 'mija', 'cariño', 'amor', 'corazón', 'querido', 'querida',
    'mamita', 'papito', 'abuelita', 'hijito', 'chiquito', 'chiquita',
    # Emotional expressions
    'triste', 'feliz', 'enojado', 'enojada', 'preocupado', 'preocupada',
    'emocionado', 'emocionada', 'nervioso', 'nerviosa', 'asustado',
    # Intensifiers
    'muy', 'tanto', 'demasiado', 'súper', 'bien', 'tan',
    # Common emotional phrases (as single words to detect)
    'dios', 'cielos', 'madre', 'verdad', 'increíble', 'horrible',
    'maravilloso', 'terrible', 'fantástico', 'lindo', 'linda', 'hermoso',
    'feo', 'loco', 'loca', 'genial', 'buenísimo', 'malísimo'
}

ENGLISH_AFFECTIVE = {
    # Interjections
    'wow', 'oh', 'ah', 'ooh', 'yay', 'ugh', 'whoa', 'gosh', 'geez',
    # Endearments
    'honey', 'sweetie', 'dear', 'darling', 'baby', 'sweetheart',
    # Emotional expressions
    'sad', 'happy', 'angry', 'worried', 'excited', 'nervous', 'scared',
    'upset', 'thrilled', 'frustrated', 'anxious', 'depressed', 'joyful',
    # Intensifiers
    'very', 'really', 'so', 'super', 'extremely', 'totally', 'absolutely',
    # Emotional adjectives
    'amazing', 'terrible', 'horrible', 'wonderful', 'fantastic', 'awful',
    'incredible', 'beautiful', 'ugly', 'crazy', 'awesome', 'great', 'bad'
}

# Neutral/functional lexicons
SPANISH_NEUTRAL = {
    'el', 'la', 'los', 'las', 'un', 'una', 'de', 'que', 'en', 'por',
    'para', 'con', 'sin', 'sobre', 'entre', 'hacia', 'desde', 'hasta',
    'ser', 'estar', 'tener', 'hacer', 'poder', 'decir', 'ir', 'ver'
}

ENGLISH_NEUTRAL = {
    'the', 'a', 'an', 'of', 'that', 'in', 'for', 'with', 'without',
    'about', 'between', 'toward', 'from', 'until', 'be', 'is', 'are',
    'have', 'do', 'make', 'can', 'say', 'go', 'see', 'get', 'take'
}

# Count affective terms by language
affective_counts = {'spanish': 0, 'english': 0}
neutral_counts = {'spanish': 0, 'english': 0}
total_by_lang = {'spanish': 0, 'english': 0}

# Track which affective words appear
affective_words_found = {'spanish': defaultdict(int), 'english': defaultdict(int)}

# Track affective usage by utterance type
utterances_with_affective = {'spanish_dominant': 0, 'english_dominant': 0}
total_utterances_by_type = {'spanish_dominant': 0, 'english_dominant': 0}

for conv in conversations:
    for utt in conv['utterances']:
        tagged = utt.get('tagged_tokens', [])
        
        # Determine dominant language
        lang_counts = {'en': 0, 'es': 0}
        for token, lang in tagged:
            if lang in lang_counts:
                lang_counts[lang] += 1
        
        dominant = 'spanish_dominant' if lang_counts['es'] > lang_counts['en'] else 'english_dominant'
        total_utterances_by_type[dominant] += 1
        
        utt_has_affective = False
        
        for token, lang in tagged:
            token_lower = token.lower().strip('.,!?¿¡')
            
            if lang == 'es':
                total_by_lang['spanish'] += 1
                if token_lower in SPANISH_AFFECTIVE:
                    affective_counts['spanish'] += 1
                    affective_words_found['spanish'][token_lower] += 1
                    utt_has_affective = True
                elif token_lower in SPANISH_NEUTRAL:
                    neutral_counts['spanish'] += 1
                    
            elif lang == 'en':
                total_by_lang['english'] += 1
                if token_lower in ENGLISH_AFFECTIVE:
                    affective_counts['english'] += 1
                    affective_words_found['english'][token_lower] += 1
                    utt_has_affective = True
                elif token_lower in ENGLISH_NEUTRAL:
                    neutral_counts['english'] += 1
        
        if utt_has_affective:
            utterances_with_affective[dominant] += 1

# Calculate metrics
print("\n--- Raw Affective Term Counts ---")
print(f"Spanish affective terms: {affective_counts['spanish']}")
print(f"English affective terms: {affective_counts['english']}")

print("\n--- Normalized Affective Rate (per 1000 tokens) ---")
es_affective_rate = (affective_counts['spanish'] / total_by_lang['spanish']) * 1000 if total_by_lang['spanish'] > 0 else 0
en_affective_rate = (affective_counts['english'] / total_by_lang['english']) * 1000 if total_by_lang['english'] > 0 else 0
print(f"Spanish: {es_affective_rate:.2f} affective terms per 1000 tokens")
print(f"English: {en_affective_rate:.2f} affective terms per 1000 tokens")
print(f"Affective Skew Ratio (ES:EN): {es_affective_rate/en_affective_rate:.2f}:1" if en_affective_rate > 0 else "N/A")

print("\n--- Affective Density by Utterance Type ---")
es_density = (utterances_with_affective['spanish_dominant'] / total_utterances_by_type['spanish_dominant'] * 100) if total_utterances_by_type['spanish_dominant'] > 0 else 0
en_density = (utterances_with_affective['english_dominant'] / total_utterances_by_type['english_dominant'] * 100) if total_utterances_by_type['english_dominant'] > 0 else 0
print(f"Spanish-dominant utterances with affective content: {utterances_with_affective['spanish_dominant']}/{total_utterances_by_type['spanish_dominant']} ({es_density:.1f}%)")
print(f"English-dominant utterances with affective content: {utterances_with_affective['english_dominant']}/{total_utterances_by_type['english_dominant']} ({en_density:.1f}%)")

print("\n--- Top 10 Spanish Affective Words ---")
for word, count in sorted(affective_words_found['spanish'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {word}: {count}")

print("\n--- Top 10 English Affective Words ---")
for word, count in sorted(affective_words_found['english'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {word}: {count}")

# ============================================================
# 2. SYNTACTIC SIMPLIFICATION ANALYSIS
# ============================================================
print("\n" + "="*70)
print("2. SYNTACTIC SIMPLIFICATION ANALYSIS")
print("="*70)

# Complexity markers
SPANISH_COMPLEXITY = {
    # Subordinate conjunctions
    'aunque', 'porque', 'mientras', 'cuando', 'donde', 'como', 'si',
    'puesto', 'dado', 'mientras', 'apenas', 'según',
    # Relative pronouns
    'quien', 'quienes', 'cual', 'cuales', 'cuyo', 'cuya', 'cuyos',
    # Complex verb forms (subjunctive markers)
    'hubiera', 'hubiese', 'pudiera', 'pudiese', 'quisiera', 'quisiese',
    'fuera', 'fuese', 'estuviera', 'estuviese', 'tuviera', 'tuviese',
    # Discourse markers
    'además', 'sin embargo', 'no obstante', 'por lo tanto', 'asimismo',
    'entonces', 'luego', 'después', 'antes', 'finalmente', 'primero'
}

ENGLISH_COMPLEXITY = {
    # Subordinate conjunctions
    'although', 'because', 'while', 'when', 'where', 'how', 'if',
    'since', 'unless', 'whereas', 'whenever', 'wherever', 'whether',
    # Relative pronouns
    'who', 'whom', 'whose', 'which', 'that',
    # Complex constructions
    'would', 'could', 'should', 'might', 'must',
    'having', 'being', 'been',
    # Discourse markers
    'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
    'consequently', 'meanwhile', 'subsequently', 'additionally', 'thus'
}

# Analyze complexity
complexity_counts = {'spanish': 0, 'english': 0}
complexity_words_found = {'spanish': defaultdict(int), 'english': defaultdict(int)}

# Track average tokens per utterance by dominant language
tokens_per_utterance = {'spanish_dominant': [], 'english_dominant': []}

# Track clause indicators
clause_counts = {'spanish': 0, 'english': 0}

for conv in conversations:
    for utt in conv['utterances']:
        tagged = utt.get('tagged_tokens', [])
        
        lang_counts = {'en': 0, 'es': 0}
        for token, lang in tagged:
            if lang in lang_counts:
                lang_counts[lang] += 1
        
        dominant = 'spanish_dominant' if lang_counts['es'] > lang_counts['en'] else 'english_dominant'
        tokens_per_utterance[dominant].append(utt.get('tokens', 0))
        
        for token, lang in tagged:
            token_lower = token.lower().strip('.,!?¿¡')
            
            if lang == 'es' and token_lower in SPANISH_COMPLEXITY:
                complexity_counts['spanish'] += 1
                complexity_words_found['spanish'][token_lower] += 1
            elif lang == 'en' and token_lower in ENGLISH_COMPLEXITY:
                complexity_counts['english'] += 1
                complexity_words_found['english'][token_lower] += 1

# Calculate metrics
print("\n--- Raw Complexity Marker Counts ---")
print(f"Spanish complexity markers: {complexity_counts['spanish']}")
print(f"English complexity markers: {complexity_counts['english']}")

print("\n--- Normalized Complexity Rate (per 1000 tokens) ---")
es_complex_rate = (complexity_counts['spanish'] / total_by_lang['spanish']) * 1000 if total_by_lang['spanish'] > 0 else 0
en_complex_rate = (complexity_counts['english'] / total_by_lang['english']) * 1000 if total_by_lang['english'] > 0 else 0
print(f"Spanish: {es_complex_rate:.2f} complexity markers per 1000 tokens")
print(f"English: {en_complex_rate:.2f} complexity markers per 1000 tokens")

simplification_index = en_complex_rate / es_complex_rate if es_complex_rate > 0 else 0
print(f"\nSyntactic Simplification Index (EN:ES): {simplification_index:.2f}")
print("(Values > 1 indicate Spanish is syntactically simpler; < 1 indicates English is simpler)")

print("\n--- Average Utterance Length ---")
avg_es = sum(tokens_per_utterance['spanish_dominant']) / len(tokens_per_utterance['spanish_dominant']) if tokens_per_utterance['spanish_dominant'] else 0
avg_en = sum(tokens_per_utterance['english_dominant']) / len(tokens_per_utterance['english_dominant']) if tokens_per_utterance['english_dominant'] else 0
print(f"Spanish-dominant utterances: {avg_es:.2f} tokens (n={len(tokens_per_utterance['spanish_dominant'])})")
print(f"English-dominant utterances: {avg_en:.2f} tokens (n={len(tokens_per_utterance['english_dominant'])})")
print(f"Length Ratio (EN:ES): {avg_en/avg_es:.2f}" if avg_es > 0 else "N/A")

print("\n--- Top 10 Spanish Complexity Markers ---")
for word, count in sorted(complexity_words_found['spanish'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {word}: {count}")

print("\n--- Top 10 English Complexity Markers ---")
for word, count in sorted(complexity_words_found['english'].items(), key=lambda x: -x[1])[:10]:
    print(f"  {word}: {count}")

# ============================================================
# 3. SUMMARY TABLE FOR REPORT
# ============================================================
print("\n" + "="*70)
print("SUMMARY TABLE FOR REPORT")
print("="*70)

print("\n+"+"-"*68+"+")
print(f"| {'Metric':<35} | {'Spanish':>12} | {'English':>12} |")
print("+"+"-"*68+"+")
print(f"| {'Total Tokens':<35} | {total_by_lang['spanish']:>12,} | {total_by_lang['english']:>12,} |")
print(f"| {'Affective Terms (raw)':<35} | {affective_counts['spanish']:>12} | {affective_counts['english']:>12} |")
print(f"| {'Affective Rate (per 1000 tokens)':<35} | {es_affective_rate:>12.2f} | {en_affective_rate:>12.2f} |")
print(f"| {'Complexity Markers (raw)':<35} | {complexity_counts['spanish']:>12} | {complexity_counts['english']:>12} |")
print(f"| {'Complexity Rate (per 1000 tokens)':<35} | {es_complex_rate:>12.2f} | {en_complex_rate:>12.2f} |")
print(f"| {'Avg Utterance Length (tokens)':<35} | {avg_es:>12.2f} | {avg_en:>12.2f} |")
print("+"+"-"*68+"+")

print(f"\n** Affective Skew Ratio (ES:EN): {es_affective_rate/en_affective_rate:.2f}:1 **" if en_affective_rate > 0 else "")
print(f"** Syntactic Simplification Index (EN:ES): {simplification_index:.2f} **")