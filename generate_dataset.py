import requests
import json
import time
import random
import csv
from datetime import datetime

# IMPORTANT: The API key is automatically provided in the Canvas environment.
API_KEY = ""
MODEL_ID = "gemini-2.5-flash-preview-05-20"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_ID}:generateContent?key={API_KEY}"

def generate_response(chat_history):
    """
    Sends a chat history to the Gemini API and returns the text response.
    Includes exponential backoff for robust API calls.
    """
    retries = 5
    delay = 1
    for i in range(retries):
        try:
            payload = {"contents": chat_history}
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()

            result = response.json()
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text
            else:
                print(f"Warning: Unexpected API response structure on attempt {i+1}.")
                raise ValueError("Unexpected API response structure.")

        except requests.exceptions.RequestException as e:
            print(f"API call failed on attempt {i+1}: {e}")
            if i < retries - 1:
                sleep_time = delay * (2 ** i) + random.uniform(0, 1)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("All retries failed. Giving up.")
                return "API ERROR: Failed to get response after multiple retries."
        except ValueError as e:
            print(f"Error parsing API response on attempt {i+1}: {e}")
            return "API ERROR: Failed to parse response."
    return None

def tag_languages(text):
    """
    Enhanced language tagging for tokens. Returns list of (token, language_tag).
    """
    import re
    
    # Expanded Spanish word list
    spanish_words = {
        'que', 'pero', 'por', 'esta', 'ese', 'esa', 'si', 'como', 'cuando',
        'donde', 'muy', 'bien', 'mal', 'hola', 'gracias', 'bueno', 'entonces',
        'porque', 'tambien', 'ya', 'ahora', 'aqui', 'alli', 'todo', 'nada',
        'algo', 'alguien', 'nadie', 'mas', 'menos', 'otro', 'otra', 'mi', 'tu',
        'su', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del',
        'al', 'y', 'o', 'con', 'sin', 'para', 'sobre', 'entre', 'hasta', 'desde',
        'fijate', 'oye', 'pues', 'verdad', 'claro', 'dime', 'sabes', 'mira',
        'ayer', 'hoy', 'mañana', 'siempre', 'nunca', 'tambien', 'despues',
        'antes', 'luego', 'tarde', 'temprano', 'vez', 'veces', 'dia', 'dias',
        'año', 'años', 'mes', 'semana', 'hora', 'tiempo', 'vida', 'gente',
        'persona', 'personas', 'cosa', 'cosas', 'trabajo', 'casa', 'ciudad',
        'pais', 'mundo', 'parte', 'lugar', 'momento', 'problema', 'manera',
        'forma', 'agua', 'comida', 'familia', 'amigo', 'amiga', 'hijo', 'hija',
        'padre', 'madre', 'hermano', 'hermana', 'hacer', 'ver', 'ir', 'venir',
        'decir', 'dar', 'poder', 'poner', 'saber', 'querer', 'estar', 'ser',
        'tener', 'haber', 'deber', 'quiero', 'tengo', 'estoy', 'soy', 'fue',
        'era', 'iba', 'hacia', 'hice', 'dijo', 'voy', 'vas', 'va', 'vamos',
        'van', 'fui', 'estas', 'estan', 'estaba', 'estuve', 'tiene', 'tienen',
        'tenia', 'tuve', 'hace', 'hacen', 'hizo', 'hicieron', 'dice', 'dicen'
    }
    
    # Tokenize preserving original case for output
    tokens_with_case = re.findall(r'\b\w+\b|[.,!?;]', text)
    tokens_lower = [t.lower() for t in tokens_with_case]
    
    tagged = []
    for original_token, lower_token in zip(tokens_with_case, tokens_lower):
        clean_token = lower_token.strip('.,!?;')
        # Check for Spanish indicators
        if clean_token in spanish_words or any(c in 'áéíóúñ¿¡' for c in clean_token):
            tagged.append((original_token, 'es'))
        else:
            tagged.append((original_token, 'en'))
    
    return tagged

def calculate_code_switch_rate(utterances):
    """
    Calculate the percentage of utterances that contain code-switching.
    """
    mixed_count = 0
    total_count = len(utterances)
    
    for utterance in utterances:
        tagged = tag_languages(utterance)
        languages = set(tag for _, tag in tagged)
        if len(languages) > 1:
            mixed_count += 1
    
    return (mixed_count / total_count * 100) if total_count > 0 else 0

def count_tokens(text):
    """Simple token counter."""
    import re
    return len(re.findall(r'\b\w+\b', text))

# ============================================================
# STRATEGY 1: Controlled Switch Point Prompting
# ============================================================
def strategy_1_controlled_switch(topic, num_turns=12):
    """
    Generate conversation with controlled switch points.
    Instructs model to switch languages at specific positions.
    """
    print(f"\n=== STRATEGY 1: Controlled Switch Point - {topic} ===")
    
    initial_prompt = f"""You are simulating a natural bilingual conversation between two friends who speak both English and Spanish fluently. 
    
Topic: {topic}

Rules:
1. Create a conversation with exactly {num_turns} turns (alternating between Speaker A and Speaker B)
2. Each speaker should code-switch naturally - mixing English and Spanish within sentences
3. Switch to Spanish after approximately every 5-7 words in some utterances
4. Use natural transitions like "you know", "pues", "so", "entonces"
5. Format each turn as "Speaker A:" or "Speaker B:" followed by the utterance
6. Make it sound like a real casual conversation between bilinguals

Start the conversation now:"""

    chat_history = [{"role": "user", "parts": [{"text": initial_prompt}]}]
    response = generate_response(chat_history)
    
    return response, chat_history

# ============================================================
# STRATEGY 2: Role-Based Conversations
# ============================================================
def strategy_2_role_based(scenario, num_turns=12):
    """
    Generate role-based bilingual conversations with specific personas.
    """
    print(f"\n=== STRATEGY 2: Role-Based - {scenario} ===")
    
    roles = {
        "friends_planning": ("Maria (grew up in Mexico, lives in US)", "Jake (American learning Spanish)"),
        "family_dinner": ("Abuela (primarily Spanish speaker)", "Grandson (bilingual, US-born)"),
        "work_colleagues": ("Ana (marketing manager, bilingual)", "Carlos (software engineer, bilingual)"),
        "shopping": ("Customer (bilingual millennial)", "Store clerk (bilingual)")
    }
    
    speaker_a, speaker_b = roles.get(scenario, ("Bilingual Person A", "Bilingual Person B"))
    
    initial_prompt = f"""Create a natural bilingual conversation (English-Spanish) with these personas:
- Speaker A: {speaker_a}
- Speaker B: {speaker_b}

Requirements:
1. Generate exactly {num_turns} turns
2. Both speakers code-switch naturally based on their background
3. Mix languages within sentences (intra-sentential switching)
4. Use discourse markers in both languages: "pero like", "so entonces", "you know pues"
5. Format: "Speaker A:" or "Speaker B:" before each turn
6. Keep it natural and conversational

Begin the conversation:"""

    chat_history = [{"role": "user", "parts": [{"text": initial_prompt}]}]
    response = generate_response(chat_history)
    
    return response, chat_history

# ============================================================
# STRATEGY 3: Topic-Based Variation
# ============================================================
def strategy_3_topic_variation(topic, tone, num_turns=10):
    """
    Generate conversations on same topic with different tones.
    """
    print(f"\n=== STRATEGY 3: Topic Variation - {topic} ({tone}) ===")
    
    initial_prompt = f"""Generate a bilingual English-Spanish conversation about: {topic}

Tone: {tone}
Number of turns: {num_turns}

Guidelines:
1. Both speakers are fluent bilinguals
2. Code-switch naturally (20-30% of utterances should mix both languages)
3. Match the {tone} tone throughout
4. Use appropriate vocabulary for the topic
5. Format each turn as "Speaker A:" or "Speaker B:"
6. Include natural fillers and discourse markers in both languages

Create the conversation:"""

    chat_history = [{"role": "user", "parts": [{"text": initial_prompt}]}]
    response = generate_response(chat_history)
    
    return response, chat_history

# ============================================================
# STRATEGY 4: Back-Translation Code-Mixing
# ============================================================
def strategy_4_back_translation(base_topic, num_turns=10):
    """
    Generate in one language, then strategically insert phrases from the other.
    """
    print(f"\n=== STRATEGY 4: Back-Translation - {base_topic} ===")
    
    # First, generate base conversation in English
    base_prompt = f"""Generate a casual conversation about {base_topic} with {num_turns} turns. 
Use simple, natural language. Format as "Speaker A:" and "Speaker B:" """
    
    chat_history = [{"role": "user", "parts": [{"text": base_prompt}]}]
    base_response = generate_response(chat_history)
    
    # Now inject Spanish code-switching
    mixing_prompt = f"""Take this English conversation and naturally insert Spanish words and phrases to create authentic code-switching:

{base_response}

Rules:
1. Replace common expressions with Spanish equivalents (like "you know" → "sabes", "but" → "pero")
2. Use Spanish for emphasis or emotional expressions
3. Keep proper nouns and technical terms in original language
4. Make 25-30% of utterances contain code-switching
5. Keep the natural flow - don't force unnatural switches

Provide the code-switched version:"""
    
    chat_history = [{"role": "user", "parts": [{"text": mixing_prompt}]}]
    mixed_response = generate_response(chat_history)
    
    return mixed_response, chat_history

# ============================================================
# STRATEGY 5: Free Generation
# ============================================================
def strategy_5_free_generation(context, num_turns=15):
    """
    Free-form natural bilingual conversation generation.
    """
    print(f"\n=== STRATEGY 5: Free Generation - {context} ===")
    
    initial_prompt = f"""Simulate a completely natural, spontaneous conversation between two bilingual friends who grew up speaking both English and Spanish.

Context: {context}
Turns: {num_turns}

Important:
1. Don't overthink it - just let the conversation flow naturally
2. Code-switch when it feels natural (roughly 25% of utterances)
3. Use both intra-sentential (within sentence) and inter-sentential (between sentences) switching
4. Include natural speech patterns: repetitions, corrections, interruptions
5. Format: "Speaker A:" and "Speaker B:"
6. Make it sound like a real conversation you'd overhear

Go:"""

    chat_history = [{"role": "user", "parts": [{"text": initial_prompt}]}]
    response = generate_response(chat_history)
    
    return response, chat_history

# ============================================================
# DATA STRUCTURING & EXPORT
# ============================================================
def parse_conversation(conversation_text):
    """
    Parse conversation into structured format with utterances.
    """
    import re
    
    utterances = []
    lines = conversation_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Match "Speaker A:" or "Speaker B:" format
        match = re.match(r'(Speaker [AB]):\s*(.+)', line, re.IGNORECASE)
        if match:
            speaker = match.group(1)
            text = match.group(2).strip()
            
            # Tag languages
            tagged = tag_languages(text)
            token_count = len([t for t, _ in tagged])
            
            utterances.append({
                'speaker': speaker,
                'text': text,
                'tokens': token_count,
                'tagged_tokens': tagged
            })
    
    return utterances

def save_dataset(all_conversations, filename='bilingual_dataset.json'):
    """
    Save structured dataset to JSON file.
    """
    dataset = {
        'metadata': {
            'language_pair': 'en-es',
            'generation_date': datetime.now().isoformat(),
            'total_conversations': len(all_conversations),
            'total_utterances': sum(len(conv['utterances']) for conv in all_conversations),
            'total_tokens': sum(sum(u['tokens'] for u in conv['utterances']) for conv in all_conversations)
        },
        'conversations': all_conversations
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Dataset saved to {filename}")
    return dataset

def save_csv_format(all_conversations, filename='bilingual_dataset.csv'):
    """
    Save dataset in CSV format for easy analysis.
    """
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['conversation_id', 'utterance_id', 'speaker', 'text', 'tokens', 'has_code_switch'])
        
        for conv_id, conv in enumerate(all_conversations):
            for utt_id, utt in enumerate(conv['utterances']):
                languages = set(tag for _, tag in utt['tagged_tokens'])
                has_cs = 'yes' if len(languages) > 1 else 'no'
                
                writer.writerow([
                    conv_id,
                    utt_id,
                    utt['speaker'],
                    utt['text'],
                    utt['tokens'],
                    has_cs
                ])
    
    print(f"✓ CSV format saved to {filename}")

# ============================================================
# MAIN GENERATION PIPELINE
# ============================================================
def main():
    """
    Main pipeline to generate complete bilingual code-switching dataset.
    Target: 15,000-20,000 tokens, 20-30% code-switching rate
    """
    
    print("=" * 70)
    print("BILINGUAL CODE-SWITCHING DATASET GENERATOR (English-Spanish)")
    print("=" * 70)
    
    all_conversations = []
    
    # Define topics and scenarios for diversity
    topics = [
        "weekend plans", "favorite movies", "cooking recipes", "travel experiences",
        "job interview experiences", "family gatherings", "sports and hobbies",
        "technology and social media", "education and school", "music and concerts"
    ]
    
    scenarios = ["friends_planning", "family_dinner", "work_colleagues", "shopping"]
    tones = ["casual", "excited", "concerned", "humorous"]
    contexts = [
        "discussing a birthday party", "talking about a recent trip",
        "planning a family reunion", "sharing gossip about mutual friends",
        "debating best restaurants in town"
    ]
    
    # STRATEGY 1: Controlled Switch Point (20 conversations)
    for i in range(20):
        topic = random.choice(topics)
        response, _ = strategy_1_controlled_switch(topic, num_turns=random.randint(10, 14))
        utterances = parse_conversation(response)
        all_conversations.append({
            'strategy': 'controlled_switch',
            'topic': topic,
            'utterances': utterances
        })
        time.sleep(1)  # Rate limiting
    
    # STRATEGY 2: Role-Based (20 conversations)
    for i in range(20):
        scenario = random.choice(scenarios)
        response, _ = strategy_2_role_based(scenario, num_turns=random.randint(10, 14))
        utterances = parse_conversation(response)
        all_conversations.append({
            'strategy': 'role_based',
            'scenario': scenario,
            'utterances': utterances
        })
        time.sleep(1)
    
    # STRATEGY 3: Topic Variation (20 conversations)
    for i in range(20):
        topic = random.choice(topics)
        tone = random.choice(tones)
        response, _ = strategy_3_topic_variation(topic, tone, num_turns=random.randint(10, 14))
        utterances = parse_conversation(response)
        all_conversations.append({
            'strategy': 'topic_variation',
            'topic': topic,
            'tone': tone,
            'utterances': utterances
        })
        time.sleep(1)
    
    # STRATEGY 4: Back-Translation (20 conversations)
    for i in range(20):
        topic = random.choice(topics)
        response, _ = strategy_4_back_translation(topic, num_turns=random.randint(10, 14))
        utterances = parse_conversation(response)
        all_conversations.append({
            'strategy': 'back_translation',
            'topic': topic,
            'utterances': utterances
        })
        time.sleep(1)
    
    # STRATEGY 5: Free Generation (20 conversations)
    for i in range(20):
        context = random.choice(contexts)
        response, _ = strategy_5_free_generation(context, num_turns=random.randint(12, 16))
        utterances = parse_conversation(response)
        all_conversations.append({
            'strategy': 'free_generation',
            'context': context,
            'utterances': utterances
        })
        time.sleep(1)
    
    # Calculate statistics
    total_utterances = sum(len(conv['utterances']) for conv in all_conversations)
    total_tokens = sum(sum(u['tokens'] for u in conv['utterances']) for conv in all_conversations)
    
    # Calculate code-switching rate
    all_utterances = [u['text'] for conv in all_conversations for u in conv['utterances']]
    cs_rate = calculate_code_switch_rate(all_utterances)
    
    # Display results
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE - DATASET STATISTICS")
    print("=" * 70)
    print(f"Total Conversations: {len(all_conversations)}")
    print(f"Total Utterances: {total_utterances}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Code-Switching Rate: {cs_rate:.2f}%")
    print(f"Target Met: {'✓ YES' if 15000 <= total_tokens <= 20000 and 20 <= cs_rate <= 30 else '✗ NO'}")
    print("=" * 70)
    
    # Save datasets
    save_dataset(all_conversations, 'bilingual_dataset.json')
    save_csv_format(all_conversations, 'bilingual_dataset.csv')
    
    print("\n✓ All files generated successfully!")
    print("\nFiles created:")
    print("  - bilingual_dataset.json (structured format)")
    print("  - bilingual_dataset.csv (analysis-ready format)")

if __name__ == "__main__":
    main()