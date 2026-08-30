import re
from datetime import datetime

# High-trust institutional entities, scientific bodies, and reputable wire agencies
HIGH_TRUST_ENTITIES = [
    r'\b(?:world health organization|who|centers for disease control|cdc|fda|nih)\b',
    r'\b(?:federal reserve|central bank|european central bank|bank of england|imf|world bank)\b',
    r'\b(?:nasa|esa|cern|noaa|usgs|national science foundation|nsf)\b',
    r'\b(?:reuters|associated press|\bap\b|bloomberg|bbc|afp|pbs|npr)\b',
    r'\b(?:harvard|mit|stanford|oxford|cambridge|johns hopkins|yale|princeton)\b',
    r'\b(?:nature journal|science journal|the lancet|new england journal of medicine|nejm|jama)\b',
    r'\b(?:supreme court|department of justice|treasury department|bureau of labor statistics|health ministry|ministry)\b',
    r'\b(?:government announced|official press release|infrastructure plan|official statement|hospital data|peer[-\s]?reviewed study)\b',
    r'\b(?:researchers published|renewable energy adoption|central bank kept interest rates)\b'
]

# Low-trust, hearsay, and unverified source markers
LOW_TRUST_SOURCES = [
    r'\b(?:anonymous sources?|unnamed (?:insiders?|officials?)|secret whistleblower)\b',
    r'\b(?:viral (?:facebook|tiktok|telegram|whatsapp|x) post|online rumors?)\b',
    r'\b(?:conspiracy theorists?|shocking video reveals|underground network)\b',
    r'\b(?:someone told me|people are saying|circulating on social media)\b'
]

# Temporal indicators
TEMPORAL_PATTERNS = [
    r'\b(?:in 20[0-2][0-9]|dated 20[0-2][0-9]|back in 20[0-2][0-9])\b',
    r'\b(?:breaking news|happening right now|just announced today|this morning)\b',
    r'\b(?:historical data from|archived report|past decade|retrospective analysis)\b'
]

def calculate_source_credibility(text: str) -> dict:
    """
    Computes Source Credibility and Temporal Awareness:
    - High-trust entity mentions (+ weight)
    - Low-trust hearsay/anonymous mentions (- weight)
    - Normalized trust score in range [-1.0, +1.0]
    - Temporal context detection
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'trust_score': 0.0,
            'trusted_entities': [],
            'low_trust_markers': [],
            'has_trusted_source': False,
            'has_hearsay': False,
            'temporal_context': 'undated'
        }

    t_lower = text.lower()

    trusted_found = []
    for pattern in HIGH_TRUST_ENTITIES:
        matches = re.findall(pattern, t_lower)
        if matches:
            trusted_found.extend(matches)

    low_trust_found = []
    for pattern in LOW_TRUST_SOURCES:
        matches = re.findall(pattern, t_lower)
        if matches:
            low_trust_found.extend(matches)

    trusted_count = len(trusted_found)
    low_trust_count = len(low_trust_found)

    # Net Trust Score calculation bounded in [-1.0, 1.0]
    raw_trust = (trusted_count * 0.40) - (low_trust_count * 0.50)
    trust_score = max(-1.0, min(1.0, round(raw_trust, 2)))

    # Temporal context analysis
    has_breaking = bool(re.search(r'\b(?:breaking news|happening right now|just in)\b', t_lower))
    has_historical = bool(re.search(r'\b(?:in 20[0-1][0-9]|historical|archived)\b', t_lower))
    
    if has_breaking:
        temporal_context = 'breaking / urgent'
    elif has_historical:
        temporal_context = 'historical archive'
    else:
        temporal_context = 'standard reportage'

    return {
        'trust_score': trust_score,
        'trusted_entities': list(set(trusted_found))[:5],
        'low_trust_markers': list(set(low_trust_found))[:5],
        'has_trusted_source': (trusted_count > 0),
        'has_hearsay': (low_trust_count > 0),
        'temporal_context': temporal_context
    }

