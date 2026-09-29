import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data dependencies are downloaded safely
def _init_nltk():
    for corpus in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab']:
        try:
            if corpus == 'stopwords':
                stopwords.words('english')
            elif corpus in ['punkt', 'punkt_tab']:
                nltk.sent_tokenize('Test sentence.')
            else:
                nltk.data.find(f'corpora/{corpus}')
        except LookupError:
            nltk.download(corpus, quiet=True)

_init_nltk()

_negation_words = {
    'not', 'no', 'never', 'neither', 'nor', 'none', 'cannot', 'without', 'against',
    'false', 'fake', 'disproven', 'misleading', 'misinterpreted', 'has', 'released',
    'true', 'fact', 'check', 'evidence', 'claim', 'alleged', 'rumor', 'hoax', 'unverified',
    'myth', 'busted', 'debunk', 'debunked'
}
_default_stopwords = set(stopwords.words('english'))
_stop_words = _default_stopwords - _negation_words

_lemmatizer = WordNetLemmatizer()

def strip_publisher_watermarks(text: str) -> str:
    """
    Comprehensive publisher, agency dateline, and attribution sanitizer.
    Eliminates shortcuts like 'Reuters', 'Via: Breitbart', and wire tags
    to prevent machine learning models from learning source bias rather than content veracity.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    t = text
    # 1. Wire Datelines: "WASHINGTON (Reuters) -", "NEW YORK (AP) —"
    t = re.sub(r'^(?:[A-Z\s,]{2,40}\s*)?\((?:Reuters|AP|AFP|Bloomberg|CNN|Fox News|BBC)\)\s*[-–—:]+\s*', '', t, flags=re.IGNORECASE)
    # 2. Bracketed/parenthetical agency markers
    t = re.sub(r'[\(\[]\s*(?:Reuters|Associated Press|AP|AFP|Bloomberg|CNN|Fox News|Breitbart|Infowars)\s*[\)\]]', '', t, flags=re.IGNORECASE)
    # 3. Trailing/inline web attribution signatures
    t = re.sub(r'\b(?:via:?|source:?|h/t:?|read more:?)\s+[A-Za-z0-9\s\.\-_&]+$', '', t, flags=re.IGNORECASE | re.MULTILINE)
    t = re.sub(r'\b(?:via|h/t)\s*[:\-]?\s*(?:Breitbart|Gateway Pundit|Daily Caller|InfoWars|Judicial Watch|Weasel Zippers|Daily Mail|WND|RT|Zero Hedge|The Blaze|True Pundit|Conservative Treehouse)\b.*', '', t, flags=re.IGNORECASE)
    # 4. URLs and social links
    t = re.sub(r'https?://\S+|www\.\S+', '', t)
    t = re.sub(r'pic\.twitter\.com/\S+|twitter\.com/\S+', '', t)
    t = re.sub(r'\[(?:Video|VIDEO|WATCH|Watch|Graphic Video)\]', '', t)
    # 5. Publisher brand tokens
    t = re.sub(r'\b(?:reuters|breitbart news|gateway pundit|daily caller|infowars|judicial watch|weasel zippers)\b', '', t, flags=re.IGNORECASE)
    return t.strip()


def clean_text(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """
    Robust NLP text cleaning pipeline:
    1. Strip news agency datelines & publisher watermarks
    2. Lowercase text
    3. Strip HTML tags, URLs, and emails
    4. Remove numbers and punctuation (except alpha characters and spaces)
    5. Tokenize by whitespace
    6. Filter English stopwords while PRESERVING negations and key fact-checking terms
    7. Lemmatize tokens
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Strip publisher datelines & agency/web artifacts (prevents memorizing 'Reuters', 'Getty', etc.)
    text_stripped = strip_publisher_watermarks(text)

    # 2. Lowercase
    text_lower = text_stripped.lower()

    # 3. Remove HTML tags, URLs, emails
    text_clean = re.sub(r'<[^>]+>', '', text_lower)
    text_clean = re.sub(r'http\S+|www\S+|https\S+', '', text_clean, flags=re.MULTILINE)
    text_clean = re.sub(r'\S+@\S+', '', text_clean)

    # 4. Remove numbers and special characters (keep letters and spaces)
    text_clean = re.sub(r'[^a-zA-Z\s]', ' ', text_clean)

    # 5. Tokenize by whitespace
    tokens = text_clean.split()

    # 6. Remove Stopwords (preserving negations & key terms for n-gram features)
    if remove_stopwords:
        tokens = [word for word in tokens if (word not in _stop_words) and (len(word) > 1)]

    # 7. Lemmatize tokens
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def detect_clickbait_indicators(text: str) -> dict:
    """
    Detects clickbait indicators:
    - Excessive exclamation/question marks (!!!, ???)
    - ALL CAPS ratio
    """
    if not isinstance(text, str) or not text.strip():
        return {'excessive_punctuation': False, 'all_caps_ratio': 0.0, 'clickbait_score': 0.0}

    # Count excessive punctuation
    exclamation_cnt = text.count('!')
    question_cnt = text.count('?')
    has_excessive_punct = (exclamation_cnt + question_cnt) >= 3

    # Calculate ALL CAPS word ratio
    words = text.split()
    caps_words = [w for w in words if w.isupper() and len(w) > 2]
    all_caps_ratio = len(caps_words) / max(len(words), 1)

    clickbait_score = min(1.0, (exclamation_cnt * 0.15 + question_cnt * 0.15 + all_caps_ratio * 0.7))

    return {
        'excessive_punctuation': has_excessive_punct,
        'all_caps_ratio': round(all_caps_ratio, 2),
        'clickbait_score': round(clickbait_score, 2)
    }

def summarize_long_text(text: str, max_words: int = 400) -> str:
    """
    Truncates long articles (> 400 words) while keeping the lead section and key assertions.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    words = text.split()
    if len(words) <= max_words:
        return text

    head = words[:300]
    tail = words[-100:]
    return " ".join(head + tail)

def extract_keywords(text: str, top_n: int = 10):
    """
    Extract top frequency keywords from cleaned text.
    """
    cleaned = clean_text(text)
    words = cleaned.split()
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
# Explicit Academic / Research / Scientific patterns
_academic_patterns = [
    r'\b(?:this work proposed|proposed (?:model|method|system|framework|approach))\b',
    r'\b(?:neural networks?|lstm|gru|recurrent neural networks?|deep learning|machine learning)\b',
    r'\b(?:ensemble technique|state of the art|benchmark|baseline)\b',
    r'\b(?:datasets?|literature|methodology|experimental results?|empirical analysis)\b',
    r'\b(?:android application|developed for|algorithm|tested on a large dataset|tested using)\b'
]

# Explicit Fact-Checking / Debunking patterns (Requires genuine debunk phrasing)
_debunk_phrases = [
    r'\b(?:fact[-\s]?check(?:ed)?|debunk(?:ed|ing)?|hoax exposed|proven false|falsely claimed|claim is (?:completely )?false|evidence shows this is false|no evidence supports this claim)\b',
    r'\b(?:independent fact-checkers? confirmed|myth busted|disproven by experts|this claim has been debunked)\b'
]

# Unverified Claim patterns (Viral rumors, conspiracy theories)
_claim_phrases = [
    r'\b(?:allegedly|it is claimed that|rumored that|sources claim|unverified reports?|conspiracy to|whistleblower claims|secretly planned)\b',
    r'\b(?:they don\'t want you to know|mainstream media won\'t tell you|secret truth|hidden cure|shocking revelation)\b'
]

# Exaggeration & Misinterpretation markers (Nuanced / Partially True News)
_exaggeration_patterns = [
    r'\b(?:social media posts? have amplified|leading to exaggerated conclusions|often misinterpreted|exaggerated claims?|amplified the findings)\b',
    r'\b(?:not yet undergone large-scale clinical validation|cautioned that the findings are still in an experimental phase|not been officially endorsed)\b',
    r'\b(?:cannot independently diagnose|taken out of context|prematurely reported|without comprehensive clinical evaluation)\b'
]

def has_exaggeration_or_misinterpretation_signals(text: str) -> bool:
    """Detects whether text discusses scientific exaggerations, media amplification, or partial truths."""
    if not isinstance(text, str) or not text.strip():
        return False
    t_lower = text.lower()
    return any(re.search(p, t_lower) for p in _exaggeration_patterns)

def is_academic_or_meta_text(text: str) -> bool:
    """Detects whether text is a pure academic abstract or scientific paper (excluding news analyzing media exaggerations)."""
    if not isinstance(text, str) or not text.strip():
        return False
    
    # If the text is specifically reporting on social media exaggerations/misinterpretations, it is a nuanced MISLEADING news review
    if has_exaggeration_or_misinterpretation_signals(text):
        return False

    t_lower = text.lower()
    matched_patterns = sum(1 for p in _academic_patterns if re.search(p, t_lower))
    return matched_patterns >= 2

def calculate_debunk_strength(text: str) -> float:
    """
    Calculates debunk strength strictly based on genuine fact-checking phrasing.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0

    # If it's an academic paper or technical report describing the field of fake news, it is not a debunk article
    if is_academic_or_meta_text(text):
        return 0.0

    t_lower = text.lower()
    matches = sum(len(re.findall(p, t_lower)) for p in _debunk_phrases)
    if matches == 0:
        return 0.0

    return min(1.0, round(matches * 0.45, 2))

def analyze_sentence_claims(text: str) -> dict:
    """
    Categorizes sentences into claims, corrections, and neutral statements with high precision.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'claim_sentences': [],
            'correction_sentences': [],
            'neutral_sentences': [],
            'claim_count': 0,
            'correction_count': 0,
            'neutral_count': 0,
            'claim_to_correction_ratio': 0.0,
            'conflicting_signals': False,
            'is_academic_meta': False
        }

    is_academic = is_academic_or_meta_text(text)
    try:
        sentences = nltk.sent_tokenize(text.strip())
    except Exception:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    claim_sentences = []
    correction_sentences = []
    neutral_sentences = []

    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue

        s_lower = s_clean.lower()
        has_claim = any(re.search(p, s_lower) for p in _claim_phrases)
        has_correction = any(re.search(p, s_lower) for p in _debunk_phrases)

        if is_academic:
            # Academic/meta research text is neutral informative content
            neutral_sentences.append(s_clean)
        elif has_correction:
            correction_sentences.append(s_clean)
        elif has_claim:
            claim_sentences.append(s_clean)
        else:
            neutral_sentences.append(s_clean)

    claim_count = len(claim_sentences)
    correction_count = len(correction_sentences)
    neutral_count = len(neutral_sentences)

    ratio = round(claim_count / max(1, correction_count), 2)
    conflicting = (claim_count > 0 and correction_count > 0)

    return {
        'claim_sentences': claim_sentences[:5],
        'correction_sentences': correction_sentences[:5],
        'neutral_sentences': neutral_sentences[:5],
        'claim_count': claim_count,
        'correction_count': correction_count,
        'neutral_count': neutral_count,
        'claim_to_correction_ratio': ratio,
        'conflicting_signals': conflicting,
        'is_academic_meta': is_academic
    }

def detect_corrections(text: str) -> int:
    """Helper returning count of correction sentences."""
    return analyze_sentence_claims(text)['correction_count']

def detect_claims(text: str) -> int:
    """Helper returning count of claim sentences."""
    return analyze_sentence_claims(text)['claim_count']
