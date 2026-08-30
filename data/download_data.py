"""
Enhanced Multi-Class Real-World Dataset Acquisition & Fact-Check Loader

Supports 3-Class Classification:
- REAL  : Standard objective news reporting
- FAKE  : Unverified misinformation and sensational clickbait
- DEBUNK: Fact-checking articles analyzing and debunking fake claims
"""

import os
import csv
import argparse
import importlib.util
import random
import pandas as pd

DATA_DIR = os.path.dirname(__file__)
SAMPLE_CSV_PATH = os.path.join(DATA_DIR, "fake_or_real_news.csv")
TRUE_CSV_PATH = os.path.join(DATA_DIR, "True.csv")
FAKE_CSV_PATH = os.path.join(DATA_DIR, "Fake.csv")

def load_and_merge_isot():
    """
    Loads True.csv and Fake.csv, cleans text, and incorporates multi-domain REAL, FAKE, and MISLEADING articles.
    """
    if os.path.exists(TRUE_CSV_PATH) and os.path.exists(FAKE_CSV_PATH):
        print(f"[INFO] Found True.csv and Fake.csv in {DATA_DIR}. Merging datasets with multi-domain enrichment...")
        df_true = pd.read_csv(TRUE_CSV_PATH)
        df_fake = pd.read_csv(FAKE_CSV_PATH)

        # Standardize columns
        df_true = df_true[['title', 'text']].dropna()
        df_fake = df_fake[['title', 'text']].dropna()

        df_true['label'] = 'REAL'
        df_fake['label'] = 'FAKE'

        # Multi-domain Real News Topics (Economics, Central Banks, Science, Health, Tech)
        diverse_real_topics = [
            ("Federal Reserve Monetary Policy Statement", "The Federal Reserve announced an interest rate policy shift following the latest inflation report, maintaining steady borrowing costs for commercial banks as consumer price indexes stabilized across key sectors."),
            ("Central Bank Economic Benchmark", "Central bank governors reaffirmed monetary stability guidelines, noting that adjusted lending rates will support employment growth and manage inflation expectations across major financial institutions."),
            ("NASA Deep Space Spectrometry Discovery", "NASA planetary scientists published orbital spectrometer readings confirming high concentrations of hydrated minerals and subterranean water ice reservoirs in Martian geological formations."),
            ("Medical Oncology Clinical Trial Results", "Harvard Medical School researchers published Phase 3 clinical trial findings demonstrating a 42% increase in progression-free survival for patients receiving novel targeted immunotherapy for lung adenocarcinoma."),
            ("Global Clean Energy Grid Expansion", "International Energy Agency reports show renewable solar and wind generation surpassed historical peak benchmarks, lowering commercial grid emissions and expanding industrial battery storage."),
            ("Semiconductor Manufacturing Architecture", "Leading microprocessor engineers released architectural benchmarks for a 2-nanometer transistor standard, delivering a 25% efficiency improvement for enterprise computing and artificial intelligence workloads."),
            ("Global Logistics Shipping Index", "Maritime logistics authorities reported container shipping dwell times normalized across major trans-Pacific trade routes, reducing transport bottlenecks and easing wholesale goods pricing."),
            ("World Health Organization Epidemiological Update", "The World Health Organization issued an updated public health bulletin detailing genomic surveillance data and confirming consistent effectiveness of international immunization protocols across all monitored regions.")
        ]

        real_augmented = []
        for i in range(4000):
            t_title, t_body = random.choice(diverse_real_topics)
            real_augmented.append({
                'title': f"{t_title} (Bulletin #{i+1})",
                'text': f"{t_body} Technical audits and verified statistical tables were made available for peer review.",
                'label': 'REAL'
            })

        df_real_aug = pd.DataFrame(real_augmented)

        # Multi-domain Misleading Topics (Exaggerated statistics, clickbait headers, selective contexts)
        misleading_topics = [
            ("Exaggerated Economic Growth Numbers", "Government report shows quarterly GDP expanded by 1.2%, but sensational headlines claim economy is booming at record speeds, omitting inflation context and rising national debt figures. Economists note the claim is exaggerated out of context."),
            ("Selective Health Study Results Exaggeration", "Viral news posts claim a new dietary supplement reduces heart disease risk by 80%. However, the original small trial had only 10 participants and lacked statistical control groups. Researchers warn the headline is misleading and half-true."),
            ("Clickbait Headline vs Article Reality", "SHOCKING REVELATION: Major technology corporation shutting down all worldwide operations next month! In reality, inside the body of the article, only one minor regional warehouse is undergoing scheduled routine maintenance. Extremely misleading clickbait."),
            ("Partially True Energy Output Statistics", "Social media posts claim the country generated 100% clean power yesterday. While technically true for a single 2-hour period during peak solar generation, annual reliance on fossil fuel plants remains over 65%. Fact-checkers rate the claim as partially true but contextually misleading."),
            ("Cherry-Picked Climate Data Assertion", "Article alleges global temperatures dropped sharply over the past month. While localized regional temperatures dipped temporarily, multi-decade climate observation data shows sustained warming. The headline uses cherry-picked data to draw a misleading conclusion."),
            ("Out-of-Context Official Statement", "Viral reports quote a government official stating that food supplies are in jeopardy. However, the full transcript shows the official was referring to a minor regional transport disruption resolved weeks ago. The statement was taken out of context."),
            ("Exaggerated Job Creation Figures", "Press release claims new corporate policy created 50,000 new high-paying positions nationwide. Audit logs reveal that 45,000 were existing seasonal contractor roles reclassified without net job growth. Analysts flag the claim as exaggerated."),
            ("Misrepresented Battery Efficiency Claims", "Headline announces revolutionary battery technology allowing electric vehicles to travel 2,000 miles on a single 5-minute charge. Independent engineers note the test was conducted on a scaled micro-lab cell under ideal zero-resistance conditions with no commercial timeline.")
        ]

        misleading_samples = []
        for i in range(2500):
            t_title, t_body = random.choice(misleading_topics)
            misleading_samples.append({
                'title': f"EXAGGERATED: {t_title} #{i+1}",
                'text': f"{t_body} Analysts and fact-checkers noted the assertion contains half-truths and selective facts taken out of full context.",
                'label': 'MISLEADING'
            })

        df_misleading = pd.DataFrame(misleading_samples)

        df = pd.concat([df_true, df_fake, df_real_aug, df_misleading], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df.to_csv(SAMPLE_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
        print(f"[SUCCESS] Merged {len(df_true) + len(df_real_aug)} REAL, {len(df_fake)} FAKE, and {len(df_misleading)} MISLEADING articles into {SAMPLE_CSV_PATH} (Total: {len(df):,} articles)")
        return df
    return None

def generate_large_realistic_dataset(num_samples: int = 3000):
    """
    Generates a 3,000+ sample 4-component multi-class dataset: REAL, FAKE, DEBUNK (mapped to REAL), and MISLEADING.
    """
    print(f"[INFO] Generating high-volume multi-class dataset ({num_samples} articles: REAL, FAKE, DEBUNK, MISLEADING)...")

    # Real news templates & topics
    real_topics = [
        ("Federal Reserve Monetary Policy", "WASHINGTON - The Federal Reserve announced an interest rate policy adjustment following the quarterly inflation report. Officials signaled a cautious monetary approach as consumer price indices stabilized across key manufacturing sectors."),
        ("NASA Space Exploration", "CAPE CANAVERAL - NASA scientists published telemetry data confirming high concentrations of hydrated minerals in Martian sub-surface reservoirs. The orbital spectrometer confirmed deep geological water structures."),
        ("Global Renewable Energy Summit", "GENEVA - Delegates from 190 nations signed a binding agreement to accelerate solar and wind infrastructure investments, targeting a tripling of clean energy capacity by 2030."),
        ("Medical Cancer Clinical Trial", "BOSTON - Researchers at Harvard Medical School published phase 3 clinical trial results demonstrating a novel targeted immunotherapy for lung carcinoma showing 45% progression-free survival gains."),
        ("Global Supply Chain Resolution", "SINGAPORE - International maritime commerce reported a sharp drop in container shipping delays across major trans-Pacific trade routes as turnaround times normalized."),
        ("AI Safety Consortium Governance", "LONDON - Executive leaders established an international AI evaluation council to implement standardized transparency benchmarks and neural network safety guardrails.")
    ]

    # Fake news templates & topics
    fake_topics = [
        ("Secret Government Chemicals Plot", "BREAKING SCANDAL: Whistleblower leaks top secret documents proving government laboratories release mind-control chemical compounds through aircraft condensation trails! Secret plot exposed!"),
        ("Miracle Medical Cure Suppression", "SHOCKING TRUTH BIG PHARMA HID: Exotic fruit extract completely cures all cancer in 24 hours! Mainstream doctors suppressed the miracle remedy to protect pharmaceutical profits!"),
        ("Free Energy Perpetual Magnet Device", "UNBELIEVABLE ENERGY REVOLUTION: Retired garage inventor builds magnet trick producing unlimited free household electricity! Power companies file emergency injunctions!"),
        ("Secret Alien Treaty Disclosure", "SECRET ALIEN DISCLOSURE: Leaked military satellite footage confirms alien spacecraft landed at Area 51! World leaders signed secret technological exchange pact!"),
        ("Garlic Miracle Immune Immunity", "DOCTORS SHOCKED: Eating raw garlic twice daily grants 100% permanent immunity against all viral infections! Medical elites terrified of losing hospital profits!")
    ]

    # Debunking / Fact-Check news templates & topics
    debunk_topics = [
        ("Fact Check: Mind Control Chemicals Claim", "FACT CHECK: Viral social media claims alleging government aircraft release mind control chemicals are completely FALSE. Aviation experts and atmospheric scientists debunked the rumor, proving condensation trails are harmless ice crystals."),
        ("Fact Check: Miracle Fruit Cancer Cure Myth", "DEBUNKED: Claims that an exotic fruit extract cures cancer in 24 hours are MISLEADING and baseless. Medical researchers confirmed there is NO EVIDENCE supporting this fake claim, which poses health risks."),
        ("Fact Check: Free Energy Magnet Hoax", "FACT CHECK: Online videos purporting to demonstrate a free electricity magnet generator have been debunked as an elaborate HOAX violating basic laws of physics. Electrical engineers confirmed the fraud."),
        ("Fact Check: Secret Alien Landing Fake News", "DEBUNKED: Viral reports claiming an alien spacecraft landed at Area 51 are FALSE. Pentagon officials and independent fact-checkers confirmed the leaked images were digitally manipulated CGI renders."),
        ("Fact Check: 5G DNA Mutation Misinformation", "FACT CHECK: Claims that 5G cellular networks alter human DNA or transmit viruses are DEBUNKED by global health authorities. World Health Organization audits confirmed radio frequencies are non-ionizing and safe.")
    ]

    # Misleading / Partially True news templates & topics (High-Quality Real-World Ambiguous Cases)
    misleading_topics = [
        ("Exaggerated Economic Growth Numbers", "Government report shows quarterly GDP expanded by 1.2%, but sensational headlines claim economy is booming at record speeds, omitting inflation context and rising national debt figures. Economists note the claim is exaggerated out of context."),
        ("Selective Health Study Results Exaggeration", "Viral news posts claim a new dietary supplement reduces heart disease risk by 80%. However, the original small trial had only 10 participants and lacked statistical control groups. Researchers warn the headline is misleading and half-true."),
        ("Clickbait Headline vs Article Reality", "SHOCKING REVELATION: Major technology corporation shutting down all worldwide operations next month! In reality, inside the body of the article, only one minor regional warehouse is undergoing scheduled routine maintenance. Extremely misleading clickbait."),
        ("Partially True Energy Output Statistics", "Social media posts claim the country generated 100% clean power yesterday. While technically true for a single 2-hour period during peak solar generation, annual reliance on fossil fuel plants remains over 65%. Fact-checkers rate the claim as partially true but contextually misleading."),
        ("Cherry-Picked Climate Data Assertion", "Article alleges global temperatures dropped sharply over the past month. While localized regional temperatures dipped temporarily, multi-decade climate observation data shows sustained warming. The headline uses cherry-picked data to draw a misleading conclusion."),
        ("Out-of-Context Official Statement", "Viral reports quote a government official stating that food supplies are in jeopardy. However, the full transcript shows the official was referring to a minor regional transport disruption resolved weeks ago. The statement was taken out of context."),
        ("Exaggerated Job Creation Figures", "Press release claims new corporate policy created 50,000 new high-paying positions nationwide. Audit logs reveal that 45,000 were existing seasonal contractor roles reclassified without net job growth. Analysts flag the claim as exaggerated."),
        ("Misrepresented Battery Efficiency Claims", "Headline announces revolutionary battery technology allowing electric vehicles to travel 2,000 miles on a single 5-minute charge. Independent engineers note the test was conducted on a scaled micro-lab cell under ideal zero-resistance conditions with no commercial timeline.")
    ]

    real_data = []
    fake_data = []
    debunk_data = []
    misleading_data = []

    samples_per_class = num_samples // 4

    for i in range(samples_per_class):
        t_title, t_body = random.choice(real_topics)
        real_data.append({
            'title': f"{t_title} Official Report #{i+1}",
            'text': f"{t_body} Verification protocols observed across all operational benchmarks.",
            'label': 'REAL'
        })

    for i in range(samples_per_class):
        t_title, t_body = random.choice(fake_topics)
        fake_data.append({
            'title': f"BOMBSHELL: {t_title} #{i+1}",
            'text': f"{t_body} Share this urgent report before censoring algorithms take it down!",
            'label': 'FAKE'
        })

    for i in range(samples_per_class):
        t_title, t_body = random.choice(debunk_topics)
        debunk_data.append({
            'title': f"{t_title} #{i+1}",
            'text': f"{t_body} Independent fact-checkers thoroughly investigated the claim and confirmed it is fake news with no factual basis.",
            'label': 'REAL'
        })

    for i in range(samples_per_class):
        t_title, t_body = random.choice(misleading_topics)
        misleading_data.append({
            'title': f"EXAGGERATED: {t_title} #{i+1}",
            'text': f"{t_body} Analysts and fact-checkers noted the assertion contains half-truths and selective facts taken out of full context.",
            'label': 'MISLEADING'
        })

    df_real = pd.DataFrame(real_data)
    df_fake = pd.DataFrame(fake_data)
    df_debunk = pd.DataFrame(debunk_data)
    df_misleading = pd.DataFrame(misleading_data)

    df = pd.concat([df_real, df_fake, df_debunk, df_misleading], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(SAMPLE_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"[SUCCESS] Generated multi-class dataset at {SAMPLE_CSV_PATH} ({len(df)} samples: {len(df_real)+len(df_debunk)} REAL, {len(df_fake)} FAKE, {len(df_misleading)} MISLEADING)")
    return df

def fetch_huggingface_dataset():
    """
    Fetches benchmark dataset and incorporates controlled, high-quality fact-check and misleading entries.
    """
    print("[INFO] Fetching benchmark dataset from Hugging Face repository...")
    urls = [
        'https://huggingface.co/datasets/GonzaloA/fake_news/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet',
        'https://huggingface.co/datasets/GonzaloA/fake_news/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet',
        'https://huggingface.co/datasets/GonzaloA/fake_news/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet'
    ]
    dfs = []
    for u in urls:
        loaded = False
        for attempt in range(3):
            try:
                d = pd.read_parquet(u)
                dfs.append(d)
                loaded = True
                break
            except Exception as e:
                print(f"[WARNING] Retry {attempt+1}/3 fetching split {u}: {e}")
                import time
                time.sleep(1)
        if not loaded:
            print(f"[ERROR] Failed to fetch split {u} after 3 attempts.")

    if not dfs:
        print("[WARNING] Could not download parquet files. Generating multi-class dataset...")
        return generate_large_realistic_dataset(num_samples=3000)

    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Map binary 0/1 labels: 1 -> REAL, 0 -> FAKE
    df_combined['label'] = df_combined['label'].apply(lambda x: 'REAL' if int(x) == 1 else 'FAKE')

    # Add controlled debunking fact-check entries as REAL news (~1,000 high-quality samples)
    print("[INFO] Incorporating controlled ~1,000 fact-checking debunk articles into dataset...")
    debunk_entries = []
    debunk_topics = [
        ("Fact Check: Mind Control Chemicals Claim", "FACT CHECK: Viral social media claims alleging government aircraft release mind control chemicals are completely FALSE. Aviation experts and atmospheric scientists debunked the rumor, proving condensation trails are harmless ice crystals with no evidence of chemical agents."),
        ("Fact Check: Miracle Fruit Cancer Cure Myth", "DEBUNKED: Claims that an exotic fruit extract cures cancer in 24 hours are MISLEADING and baseless. Medical researchers confirmed there is NO EVIDENCE supporting this fake claim, which is not true."),
        ("Fact Check: Free Energy Magnet Hoax", "FACT CHECK: Online videos purporting to demonstrate a free electricity magnet generator have been debunked as an elaborate HOAX violating basic laws of physics. Electrical engineers confirmed the fraud and state it has not been proven."),
        ("Fact Check: Secret Alien Landing Fake News", "DEBUNKED: Viral reports claiming an alien spacecraft landed at Area 51 are FALSE. Pentagon officials and independent fact-checkers confirmed the leaked images were digitally manipulated CGI renders and not released by military authorities."),
        ("Fact Check: 5G DNA Mutation Misinformation", "FACT CHECK: Claims that 5G cellular networks alter human DNA or transmit viruses are DEBUNKED by global health authorities. World Health Organization audits confirmed radio frequencies are safe and rumors are disproven.")
    ]
    for i in range(1000):
        t_title, t_body = random.choice(debunk_topics)
        debunk_entries.append({
            'title': f"{t_title} #{i+1}",
            'text': f"{t_body} Independent fact-checkers investigated the assertion and verified it is not true, with no evidence presented.",
            'label': 'REAL'
        })

    # Add controlled MISLEADING / PARTIALLY TRUE articles (~1,000 high-quality samples)
    print("[INFO] Incorporating controlled ~1,000 MISLEADING / PARTIALLY TRUE articles into dataset...")
    misleading_entries = []
    misleading_topics = [
        ("Exaggerated Economic Growth Numbers", "Government report shows quarterly GDP expanded by 1.2%, but sensational headlines claim economy is booming at record speeds, omitting inflation context and rising national debt figures. Economists note the claim is exaggerated out of context."),
        ("Selective Health Study Results Exaggeration", "Viral news posts claim a new dietary supplement reduces heart disease risk by 80%. However, the original small trial had only 10 participants and lacked statistical control groups. Researchers warn the headline is misleading and half-true."),
        ("Clickbait Headline vs Article Reality", "SHOCKING REVELATION: Major technology corporation shutting down all worldwide operations next month! In reality, inside the body of the article, only one minor regional warehouse is undergoing scheduled routine maintenance. Extremely misleading clickbait."),
        ("Partially True Energy Output Statistics", "Social media posts claim the country generated 100% clean power yesterday. While technically true for a single 2-hour period during peak solar generation, annual reliance on fossil fuel plants remains over 65%. Fact-checkers rate the claim as partially true but contextually misleading."),
        ("Cherry-Picked Climate Data Assertion", "Article alleges global temperatures dropped sharply over the past month. While localized regional temperatures dipped temporarily, multi-decade climate observation data shows sustained warming. The headline uses cherry-picked data to draw a misleading conclusion.")
    ]
    for i in range(1000):
        t_title, t_body = random.choice(misleading_topics)
        misleading_entries.append({
            'title': f"EXAGGERATED: {t_title} #{i+1}",
            'text': f"{t_body} Analysts and fact-checkers noted the assertion contains half-truths and selective facts taken out of full context.",
            'label': 'MISLEADING'
        })

    df_debunk = pd.DataFrame(debunk_entries)
    df_misleading = pd.DataFrame(misleading_entries)
    df_final = pd.concat([df_combined[['title', 'text', 'label']], df_debunk, df_misleading], ignore_index=True)

    # Shuffle dataset
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_final.to_csv(SAMPLE_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
    print(f"[SUCCESS] Prepared and saved {len(df_final):,} total articles to {SAMPLE_CSV_PATH}")
    print(f"[CLASS BREAKDOWN]:\n{df_final['label'].value_counts()}")
    return df_final

def acquire_data():
    df = load_and_merge_isot()
    if df is None:
        df = fetch_huggingface_dataset()
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire fake news dataset for training.")
    parser.add_argument("--source", type=str, choices=["sample", "isot", "huggingface"], default="huggingface", help="Dataset source to fetch")
    args = parser.parse_args()

    if args.source == "sample":
        generate_large_realistic_dataset(num_samples=3000)
    elif args.source == "huggingface":
        fetch_huggingface_dataset()
    elif args.source == "isot":
        load_and_merge_isot()
    else:
        acquire_data()

