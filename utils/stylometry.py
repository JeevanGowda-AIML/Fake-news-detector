import re
import numpy as np

# Stylometric lexicons
EMOTIONAL_CLICKBAIT_WORDS = {
    'shocking', 'unbelievable', 'bombshell', 'exposed', 'furious', 'destroyed',
    'screaming', 'miracle', 'terrified', 'panic', 'shredded', 'must see', 'viral',
    'breathtaking', 'horrifying', 'disaster', 'mind-blowing', 'mind-boggling'
}

FORMAL_JOURNALISTIC_VERBS = {
    'announced', 'stated', 'reported', 'confirmed', 'published', 'reaffirmed',
    'indicated', 'noted', 'concluded', 'demonstrated', 'clarified', 'documented'
}

def extract_stylometric_features(text: str) -> dict:
    """
    Extracts 10 normalized, regularized stylometric and journalistic credibility features:
    All features are strictly normalized and bounded in [0.0, 1.0] to prevent overfitting.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'sensationalism_score': 0.0,
            'all_caps_ratio': 0.0,
            'formality_score': 0.5,
            'emotional_intensity': 0.0,
            'numeric_density': 0.0,
            'quote_density': 0.0,
            'academic_score': 0.0,
            'hearsay_score': 0.0,
            'average_word_length': 0.0,
            'sentence_complexity': 0.5
        }

    words = text.split()
    total_words = max(len(words), 1)
    t_lower = text.lower()
    lower_words = [w.strip('.,!?"\'') for w in t_lower.split()]

    # 1. Sensationalism score (exclamation marks, question marks, clickbait adjectives)
    exclamations = text.count('!')
    questions = text.count('?')
    clickbait_matches = sum(1 for w in lower_words if w in EMOTIONAL_CLICKBAIT_WORDS)
    raw_sensationalism = (exclamations * 0.20) + (questions * 0.10) + (clickbait_matches / (total_words / 20.0))
    sensationalism_score = float(np.clip(raw_sensationalism, 0.0, 1.0))

    # 2. ALL CAPS Ratio
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    all_caps_ratio = float(np.clip(len(caps_words) / max(total_words / 5.0, 1.0), 0.0, 1.0))

    # 3. Formality & Objectivity Score (Journalistic verbs, third-person declarative tone)
    formal_verb_matches = sum(1 for w in lower_words if w in FORMAL_JOURNALISTIC_VERBS)
    formality_score = float(np.clip((formal_verb_matches / (total_words / 25.0)) * 0.70 + 0.30, 0.0, 1.0))

    # 4. Emotional Intensity
    emotional_matches = sum(1 for w in lower_words if w in EMOTIONAL_CLICKBAIT_WORDS)
    emotional_intensity = float(np.clip(emotional_matches / (total_words / 15.0), 0.0, 1.0))

    # 5. Numeric / Statistical Density (e.g., percentages, dates, dollar amounts, metrics)
    numeric_tokens = len(re.findall(r'\b\d+(?:\.\d+)?%?|\$\d+', text))
    numeric_density = float(np.clip(numeric_tokens / (total_words / 20.0), 0.0, 1.0))

    # 6. Direct Quote & Attribution Density
    quotes = len(re.findall(r'["\'].+?["\']', text))
    quote_density = float(np.clip(quotes / max(total_words / 40.0, 1.0), 0.0, 1.0))

    # 7. Academic / Scientific Methodology score
    academic_matches = len(re.findall(r'\b(?:proposed|dataset|neural|algorithm|empirical|methodology|literature|results)\b', t_lower))
    academic_score = float(np.clip(academic_matches / (total_words / 20.0), 0.0, 1.0))

    # 8. Hearsay / Anonymous attribution score
    hearsay_matches = len(re.findall(r'\b(?:sources claim|allegedly|insiders say|secret plot|anonymous)\b', t_lower))
    hearsay_score = float(np.clip(hearsay_matches / max(total_words / 30.0, 1.0), 0.0, 1.0))

    # 9. Average word length (normalized around standard 5-6 char words)
    avg_len = sum(len(w) for w in lower_words) / total_words
    avg_word_length = float(np.clip((avg_len - 3.0) / 5.0, 0.0, 1.0))

    # 10. Sentence complexity (sentence length variation)
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    avg_sent_len = total_words / max(len(valid_sentences), 1)
    sentence_complexity = float(np.clip(avg_sent_len / 35.0, 0.0, 1.0))

    return {
        'sensationalism_score': round(sensationalism_score, 4),
        'all_caps_ratio': round(all_caps_ratio, 4),
        'formality_score': round(formality_score, 4),
        'emotional_intensity': round(emotional_intensity, 4),
        'numeric_density': round(numeric_density, 4),
        'quote_density': round(quote_density, 4),
        'academic_score': round(academic_score, 4),
        'hearsay_score': round(hearsay_score, 4),
        'average_word_length': round(avg_word_length, 4),
        'sentence_complexity': round(sentence_complexity, 4)
    }

