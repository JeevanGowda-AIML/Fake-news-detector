import os
import pandas as pd
from utils.preprocess import clean_text

def load_and_prepare_data(file_path: str = None):
    """
    Loads multi-class dataset (REAL, FAKE, DEBUNK), cleans text, and ensures consistent format.
    Returns: X_raw, X_clean, y
    """
    if file_path is None or not os.path.exists(file_path):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(base_dir, "data", "fake_or_real_news.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please run data/download_data.py first.")

    df = pd.read_csv(file_path)

    # Standardize column names
    col_mapping = {}
    for col in df.columns:
        c_lower = col.lower().strip()
        if c_lower in ['text', 'content', 'article', 'news', 'body']:
            col_mapping[col] = 'text'
        elif c_lower in ['label', 'target', 'class', 'category']:
            col_mapping[col] = 'label'
    df = df.rename(columns=col_mapping)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Dataset at {file_path} must contain 'text' and 'label' columns.")

    df = df.dropna(subset=['text', 'label']).copy()

    # Standardize label strings into multi-class target classes: REAL, FAKE, or MISLEADING
    def normalize_label(l):
        s = str(l).upper().strip()
        if s in ['MISLEADING', 'PARTIALLY_TRUE', 'HALF_TRUE', 'EXAGGERATED', 'CLICKBAIT', 'MIXED']:
            return 'MISLEADING'
        elif s in ['DEBUNK', 'DEBUNKED', 'FACTCHECK', 'FACT-CHECK', 'HOAX-EXPOSED', '1', 'REAL', 'TRUE', 'FACT', 'GENUINE']:
            return 'REAL'
        elif s in ['0', 'FAKE', 'FALSE', 'LIE', 'MISINFORMATION']:
            return 'FAKE'
        return s

    df['label'] = df['label'].apply(normalize_label)
    df['clean_text'] = df['text'].apply(lambda t: clean_text(str(t)))

    return df['text'], df['clean_text'], df['label']
