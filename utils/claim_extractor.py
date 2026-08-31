import re
import nltk

def _init_sent_tokenizer():
    for pkg in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

_init_sent_tokenizer()

# Specific regex patterns for claim statements
CLAIM_PATTERNS = [
    r'\b(?:allegedly|it is claimed that|rumored that|sources claim|unverified reports?|conspiracy to|whistleblower claims?|whistleblower leaked|secretly planned)\b',
    r'\b(?:they don\'t want you to know|mainstream media won\'t tell you|secret truth|hidden cure|shocking revelation)\b',
    r'\b(?:claims? that|alleges? that|purports? to|said without evidence|falsely claimed|viral posts? claim|viral rumors? claims?|viral rumors?)\b',
    r'\b(?:mind[-\s]?control|microchips? in vaccines?|5g radiation kills|chem[-\s]?trails|cure all diseases?)\b',
    r'\b(?:hiding truth|destroy economy|shocking|secret plot|sources suggest)\b'
]

# Specific regex patterns for correction/debunking statements
CORRECTION_PATTERNS = [
    r'\b(?:fact[-\s]?check(?:ed)?|debunk(?:ed|ing)?|hoax exposed|proven false|falsely claimed|claim is (?:completely )?false)\b',
    r'\b(?:evidence shows this is false|no evidence supports this claim|independent fact-checkers? confirmed|myth busted)\b',
    r'\b(?:disproven by experts|this claim has been debunked|official records contradict|scientists clarified)\b',
    r'\b(?:contrary to (?:viral )?claims|misleading assertion|out of context)\b'
]

# Institutional & Factual declarative markers
FACTUAL_PATTERNS = [
    r'\b(?:government announced|infrastructure plan|official press release|ministry confirmed|department stated|public sector)\b',
    r'\b(?:announced|published|confirmed by|reported by|official statement|according to (?:the )?(?:CDC|WHO|NASA|Federal Reserve|Department|Ministry|University|Institute|Government))\b',
    r'\b(?:experts?|economists?|scientists?|researchers?|officials?|analysts?)\s+(?:say|warn|state|stated|reported|confirmed|noted|clarified|found|published)\b',
    r'\b(?:phase \d+ clinical trial|peer[-\s]?reviewed|empirical data|statistical analysis|quarterly report|survey data|hospital data)\b',
    r'\b(?:researchers published findings?|renewable energy adoption|increased by \d+%\b|subterranean water ice|subterranean)\b',
    r'\b(?:this work proposed|proposed model|neural network|algorithm|benchmark dataset)\b'
]

# Contrastive Caveat & Half-Truth Patterns (Claims paired with limitations / missing context)
CONTRASTIVE_CAVEAT_PATTERNS = [
    r'\b(?:without (?:scientific|clinical|empirical|medical|official) (?:trials?|validation|testing|evidence|proof|peer[-\s]?review))\b',
    r'\b(?:unverified (?:miraculous|medical|health|experimental) (?:cures?|treatments?|remedies|results|claims?))\b',
    r'\b(?:celebrities (?:are )?investing in unverified|celebrities claim miraculous|suggests? rapid results without)\b',
    r'\b(?:but (?:no official data|no official validation|no clinical validation|long[-\s]?term effects? are still unknown|experts warn results vary|viral posts? exaggerate|results vary widely|practical implementation remains uncertain))\b',
    r'\b(?:though no (?:official|scientific|empirical|clinical) (?:validation|evidence|data|proof|endorsement) exists?)\b',
    r'\b(?:yet no (?:official|scientific|empirical) (?:data|evidence|validation|proof) has been released)\b',
    r'\b(?:preliminary findings?|experimental phase|yet to be verified|unconfirmed reports?|not yet undergone large-scale|not officially endorsed)\b',
    r'\b(?:cannot independently diagnose|taken out of context|prematurely reported|without comprehensive clinical evaluation|leading to exaggerated conclusions)\b',
    r'\b(?:highlights? early success.*?,\s*but\b|suggests?.*?,\s*but\b|claims?.*?,\s*but\b|reports? suggest.*?,\s*though\b)'
]

def has_contrastive_caveat(text: str) -> bool:
    """Detects whether text contains a contrastive caveat, limitation, or exaggerated half-truth."""
    if not isinstance(text, str) or not text.strip():
        return False
    t_lower = text.lower()
    return any(re.search(p, t_lower) for p in CONTRASTIVE_CAVEAT_PATTERNS)

def extract_and_classify_sentences(text: str) -> dict:
    """
    Sentence-level Claim Extraction & Dissection:
    1. Splits text into discrete sentences.
    2. Tags each sentence as 'CLAIM', 'CORRECTION', 'FACTUAL', or 'NEUTRAL'.
    3. Calculates sentence contradiction tension and claim-to-correction ratio.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'sentences': [],
            'claim_sentences': [],
            'correction_sentences': [],
            'factual_sentences': [],
            'neutral_sentences': [],
            'claim_count': 0,
            'correction_count': 0,
            'factual_count': 0,
            'neutral_count': 0,
            'contradiction_score': 0.0,
            'claim_to_correction_ratio': 0.0,
            'has_contradiction': False
        }

    try:
        raw_sentences = nltk.sent_tokenize(text.strip())
    except Exception:
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

    sentence_entries = []
    claim_sents = []
    correction_sents = []
    factual_sents = []
    neutral_sents = []

    for idx, sent in enumerate(raw_sentences):
        s_clean = sent.strip()
        if not s_clean:
            continue

        s_lower = s_clean.lower()
        is_claim = any(re.search(p, s_lower) for p in CLAIM_PATTERNS)
        is_correction = any(re.search(p, s_lower) for p in CORRECTION_PATTERNS)
        is_factual = any(re.search(p, s_lower) for p in FACTUAL_PATTERNS)

        is_caveat = has_contrastive_caveat(s_clean)

        if is_correction:
            role = 'CORRECTION'
            correction_sents.append(s_clean)
        elif is_caveat or (is_factual and is_claim):
            role = 'MISLEADING'
            claim_sents.append(s_clean)
            factual_sents.append(s_clean)
        elif is_claim:
            role = 'CLAIM'
            claim_sents.append(s_clean)
        elif is_factual:
            role = 'FACTUAL'
            factual_sents.append(s_clean)
        else:
            role = 'NEUTRAL'
            neutral_sents.append(s_clean)

        sentence_entries.append({
            'index': idx,
            'text': s_clean,
            'role': role
        })

    claim_count = len(claim_sents)
    correction_count = len(correction_sents)
    factual_count = len(factual_sents)
    neutral_count = len(neutral_sents)

    # Contradiction Score (NLI tension: Coexistence of unverified claims + explicit refutations or factual assertions)
    has_contradiction = (claim_count > 0 and (correction_count > 0 or factual_count > 0))
    if correction_count > 0 and claim_count > 0:
        overlap = min(claim_count, correction_count)
        total = max(claim_count + correction_count, 1)
        contradiction_score = round(min(1.0, (overlap * 2.0) / total), 2)
    elif factual_count > 0 and claim_count > 0:
        # Mixed factual reporting with unverified viral rumor
        contradiction_score = 0.60
    else:
        contradiction_score = 0.0

    ratio = round(claim_count / max(1, correction_count), 2)

    return {
        'sentences': sentence_entries,
        'claim_sentences': claim_sents,
        'correction_sentences': correction_sents,
        'factual_sentences': factual_sents,
        'neutral_sentences': neutral_sents,
        'claim_count': claim_count,
        'correction_count': correction_count,
        'factual_count': factual_count,
        'neutral_count': neutral_count,
        'contradiction_score': contradiction_score,
        'claim_to_correction_ratio': ratio,
        'has_contradiction': has_contradiction
    }
