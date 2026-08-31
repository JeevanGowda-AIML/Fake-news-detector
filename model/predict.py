import os
import re
import joblib
import numpy as np
from utils.preprocess import clean_text, is_academic_or_meta_text
from utils.claim_extractor import extract_and_classify_sentences
from utils.source_credibility import calculate_source_credibility
from utils.knowledge_base import match_knowledge_grounding
from utils.stylometry import extract_stylometric_features


def calibrate_and_rebalance_probabilities(raw_probs: dict, temperature: float = 1.5) -> dict:
    """
    Strict 3-Step Probability Calibration Pipeline:
    Step 1: Convert raw probabilities to logits & apply Temperature Scaling (T=1.5).
    Step 2: Apply Class Rebalancing:
            - REAL: 1.10x boost
            - FAKE: 0.90x attenuation (curbs false-alarm fake bias)
            - MISLEADING: 1.25x boost (restores minority representation)
    Step 3: Normalize so sum strictly equals 1.0.
    """
    labels = ["REAL", "FAKE", "MISLEADING"]
    p_vec = np.array([max(1e-9, raw_probs.get(k, 0.3333)) for k in labels], dtype=float)

    # 1. Temperature Scaling Softmax
    logits = np.log(p_vec)
    scaled_logits = logits / max(0.1, float(temperature))
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
    scaled_p = exp_logits / np.sum(exp_logits)

    # 2. Class Rebalancing
    p_real = scaled_p[0] * 1.10
    p_fake = scaled_p[1] * 0.90
    p_misleading = scaled_p[2] * 1.25

    total = p_real + p_fake + p_misleading
    if total > 0:
        return {
            "REAL": round(float(p_real / total), 4),
            "FAKE": round(float(p_fake / total), 4),
            "MISLEADING": round(float(p_misleading / total), 4)
        }
    return {"REAL": 0.3333, "FAKE": 0.3333, "MISLEADING": 0.3334}


def calculate_misleading_score(
    claim_count: int,
    correction_count: int,
    ratio: float,
    contradiction_score: float,
    stylometry_info: dict,
    source_info: dict
) -> float:
    """
    Calculates explicit MISLEADING score in range [0.0, 1.0]:
    - 0.40 * claim_ratio_factor (normalized)
    - 0.30 * contradiction_score (NLI claim vs refutation tension)
    - 0.20 * exaggeration_score (sensational clickbait adjectives)
    - 0.10 * uncertainty_score (hearsay vs lack of primary source)
    """
    claim_factor = min(1.0, ratio / 2.0) if claim_count > 0 else 0.0
    contradiction_factor = float(contradiction_score)
    exaggeration_factor = float(stylometry_info.get('sensationalism_score', 0.0))
    
    # Uncertainty factor: presence of anonymous hearsay or lack of trusted source
    if source_info.get('has_hearsay'):
        uncertainty_factor = 0.8
    elif not source_info.get('has_trusted_source'):
        uncertainty_factor = 0.4
    else:
        uncertainty_factor = 0.1

    score = (
        0.40 * claim_factor +
        0.30 * contradiction_factor +
        0.20 * exaggeration_factor +
        0.10 * uncertainty_factor
    )
    return round(float(np.clip(score, 0.0, 1.0)), 4)


def resolve_hierarchical_decision(
    raw_probs: dict,
    sentence_info: dict,
    source_info: dict,
    knowledge_info: dict,
    stylometry_info: dict,
    text: str
) -> dict:
    """
    Expert Hierarchical Decision Engine:
    1. Pre-Calibrated & Rebalanced Probabilities
    2. Explicit Misleading Score Computation
    3. Layer 1: Academic & Scientific Literature
    4. Layer 2: Debunk / Fact-Check Logic
    5. Layer 3: Strong Rule (Mixed Claims + Corrections -> MISLEADING)
    6. Layer 4: FAKE vs MISLEADING Override (Exaggerated partial truth)
    7. Layer 5: Offline Knowledge Grounding & Source Authority
    8. Layer 6: Uncertainty Zone Guard (max_prob < 0.50 -> UNCERTAIN)
    9. Rule Confidence Injection & Final Normalization
    """
    decision_path = []

    # Step 1 & 2: Temperature Scaling & Class Rebalancing
    calibrated_probs = calibrate_and_rebalance_probabilities(raw_probs, temperature=1.5)
    p_real = calibrated_probs["REAL"]
    p_fake = calibrated_probs["FAKE"]
    p_misleading = calibrated_probs["MISLEADING"]
    decision_path.append(f"Calibrated & Rebalanced Probs: REAL={p_real:.1%}, FAKE={p_fake:.1%}, MIS={p_misleading:.1%}")

    claim_count = sentence_info.get("claim_count", 0)
    correction_count = sentence_info.get("correction_count", 0)
    ratio = sentence_info.get("claim_to_correction_ratio", 0.0)
    contradiction_score = sentence_info.get("contradiction_score", 0.0)

    trust_score = source_info.get("trust_score", 0.0)
    has_trusted = source_info.get("has_trusted_source", False)
    has_hearsay = source_info.get("has_hearsay", False)

    sensationalism = stylometry_info.get("sensationalism_score", 0.0)

    # Compute Explicit Misleading Score
    misleading_score = calculate_misleading_score(
        claim_count=claim_count,
        correction_count=correction_count,
        ratio=ratio,
        contradiction_score=contradiction_score,
        stylometry_info=stylometry_info,
        source_info=source_info
    )
    decision_path.append(f"Computed Misleading Score: {misleading_score:.2f} (contradiction={contradiction_score:.2f}, sensationalism={sensationalism:.2f})")

    final_label = None
    confidence = 0.50

    # -------------------------------------------------------------
    # LAYER 1: CONTRASTIVE CAVEAT CLAUSES & MEDIA EXAGGERATION
    # -------------------------------------------------------------
    from utils.preprocess import has_exaggeration_or_misinterpretation_signals
    from utils.claim_extractor import has_contrastive_caveat

    has_caveat = has_contrastive_caveat(text)

    if has_caveat or has_exaggeration_or_misinterpretation_signals(text):
        final_label = "MISLEADING / PARTIALLY TRUE"
        confidence = 0.85
        decision_path.append("Layer 1: Contrastive Caveat / Media Exaggeration detected: Preliminary study or claim accompanied by contextual limitations -> Resolved to MISLEADING.")
    elif is_academic_or_meta_text(text):
        final_label = "REAL"
        confidence = 0.94
        decision_path.append("Layer 1: Peer-level academic & scientific abstract detected (methodology, datasets, neural architectures).")

    # -------------------------------------------------------------
    # LAYER 2: DEBUNK / FACT-CHECK LOGIC
    # -------------------------------------------------------------
    debunk_strength = 0.85 if correction_count >= 1 else 0.0
    if final_label is None and debunk_strength >= 0.40:
        if correction_count > 0 and claim_count == 0:
            final_label = "REAL (Debunk)"
            confidence = 0.85
            decision_path.append(f"Layer 2: Dedicated Fact-Check Passed: {correction_count} refutation(s) with zero unverified claims.")
        else:
            final_label = "MISLEADING / PARTIALLY TRUE"
            confidence = 0.80
            decision_path.append(f"Layer 2: Conflicting claim vs refutation signal: {claim_count} claim(s) vs {correction_count} refutation(s) -> MISLEADING.")

    # -------------------------------------------------------------
    # LAYER 3: STRONG MIXED SIGNAL & CONTRADICTION RULE
    # -------------------------------------------------------------
    factual_count = sentence_info.get("factual_count", 0)

    # 1. Fact-checks refuting claims -> MISLEADING
    if final_label is None and claim_count > 0 and correction_count > 0:
        final_label = "MISLEADING / PARTIALLY TRUE"
        confidence = 0.80
        decision_path.append(f"Layer 3: Strong Mixed Signal: Coexistence of unverified claims ({claim_count}) and fact-check refutations ({correction_count}) -> Resolved to MISLEADING.")

    # 2. Genuine factual statements mixed with unverified viral rumors -> MISLEADING
    elif final_label is None and factual_count > 0 and claim_count > 0:
        final_label = "MISLEADING / PARTIALLY TRUE"
        confidence = 0.78
        decision_path.append(f"Layer 3: Mixed Truth + Rumor: Objective factual assertions ({factual_count}) blended with unverified viral claims ({claim_count}) -> Resolved to MISLEADING.")

    # -------------------------------------------------------------
    # LAYER 4: OFFLINE FACTUAL GROUNDING & TRUSTED SOURCE AUTHORITY
    # -------------------------------------------------------------
    if final_label is None and knowledge_info is not None:
        grounded_label = knowledge_info.get("grounded_label")
        topic = knowledge_info.get("topic")
        if grounded_label == "REAL":
            final_label = "REAL"
            confidence = 0.95
            decision_path.append(f"Layer 4: Factual Grounding: Aligns with verified ground truth on '{topic}'.")
        elif grounded_label == "FAKE":
            final_label = "FAKE"
            confidence = 0.95
            decision_path.append(f"Layer 4: Known Hoax Registry: Matched debunked hoax pattern on '{topic}'.")

    # High-trust institutional or government press release authority
    if final_label is None and has_trusted and trust_score >= 0.30 and sensationalism <= 0.25:
        final_label = "REAL"
        confidence = 0.90
        decision_path.append(f"Layer 4: High-trust institutional source authority detected (Trust Score: +{trust_score:.2f}).")

    # Empirical Research Declarative Fallback (e.g. "Researchers published findings showing ... increased by 15%")
    if final_label is None and factual_count > 0 and claim_count == 0 and sensationalism <= 0.15:
        final_label = "REAL"
        confidence = 0.85
        decision_path.append(f"Layer 4: Empirical Declarative Finding: Objective research publication with {factual_count} factual statement(s) and zero sensationalism.")

    # -------------------------------------------------------------
    # LAYER 5: MISLEADING VS FAKE BOUNDARY (Requires Mixed Context or Explicit Caveat)
    # -------------------------------------------------------------
    if final_label is None:
        # Misleading requires an actual counter-signal, caveat, or mixed truth
        is_mixed_truth_or_caveat = (has_caveat or (factual_count > 0 and claim_count > 0) or (correction_count > 0 and claim_count > 0))

        if p_fake > 0.60 and is_mixed_truth_or_caveat:
            final_label = "MISLEADING / PARTIALLY TRUE"
            confidence = max(0.75, misleading_score)
            decision_path.append(f"Layer 5: Overrode statistical FAKE -> MISLEADING: text contains partial truth, limitation, or caveat clause.")
        elif ratio > 1.2 and claim_count > 0 and factual_count > 0:
            final_label = "MISLEADING / PARTIALLY TRUE"
            confidence = 0.72
            decision_path.append(f"Layer 5: Claim ratio ({ratio:.2f}) indicates claims outnumber factual context -> MISLEADING.")

    # -------------------------------------------------------------
    # LAYER 6: CALIBRATED STATISTICAL MODEL PROBABILITY
    # -------------------------------------------------------------
    if final_label is None:
        if p_fake >= 0.65:
            final_label = "FAKE"
            confidence = p_fake
            decision_path.append(f"Layer 6: Calibrated statistical probability indicates FAKE (P={p_fake:.2%}).")
        elif p_real >= 0.65:
            final_label = "REAL"
            confidence = p_real
            decision_path.append(f"Layer 6: Calibrated statistical probability indicates REAL (P={p_real:.2%}).")
        elif p_misleading >= 0.45:
            final_label = "MISLEADING / PARTIALLY TRUE"
            confidence = p_misleading
            decision_path.append(f"Layer 6: Calibrated statistical probability indicates MISLEADING (P={p_misleading:.2%}).")

    # -------------------------------------------------------------
    # LAYER 7: UNCERTAINTY ZONE SAFETY GUARD (max_prob < 0.50)
    # -------------------------------------------------------------
    if final_label is None or max(p_real, p_fake, p_misleading) < 0.50:
        final_label = "HIGHLY UNCERTAIN"
        confidence = 0.45
        decision_path.append("Layer 7: Ambiguous input with low confidence margin (< 50%). Routed to Uncertainty Zone.")

    # -------------------------------------------------------------
    # STEP 4 & 5: INJECT RULE CONFIDENCE & FINAL NORMALIZATION
    # -------------------------------------------------------------
    p_arr = [p_real, p_fake, p_misleading]  # Index 0: REAL, 1: FAKE, 2: MISLEADING

    if "MISLEADING" in final_label:
        p_arr[2] = max(p_arr[2], 0.65)
        rem = 1.0 - p_arr[2]
        p_arr[0] = rem * 0.50
        p_arr[1] = rem * 0.50
    elif "REAL (Debunk)" in final_label:
        p_arr[0] = max(p_arr[0], 0.80)
        rem = 1.0 - p_arr[0]
        p_arr[1] = rem * 0.70
        p_arr[2] = rem * 0.30
    elif final_label == "REAL":
        p_arr[0] = max(p_arr[0], 0.80)
        rem = 1.0 - p_arr[0]
        p_arr[1] = rem * 0.70
        p_arr[2] = rem * 0.30
    elif final_label == "FAKE":
        p_arr[1] = max(p_arr[1], 0.80)
        rem = 1.0 - p_arr[1]
        p_arr[0] = rem * 0.70
        p_arr[2] = rem * 0.30
    else:  # UNCERTAIN
        p_arr = [0.34, 0.33, 0.33]

    # Final Normalization strictly guaranteeing sum = 1.0
    tot = sum(p_arr)
    final_probs = {
        "REAL": round(p_arr[0] / tot, 4),
        "FAKE": round(p_arr[1] / tot, 4),
        "MISLEADING": round(p_arr[2] / tot, 4)
    }

    confidence = float(np.clip(confidence, 0.0, 1.0))

    return {
        "final_label": final_label,
        "confidence": round(confidence, 4),
        "probabilities": final_probs,
        "misleading_score": misleading_score,
        "decision_path": decision_path
    }


class FakeNewsPredictor:
    """
    Expert-System Production Misinformation Analyzer:
    - 6-Layer Hierarchical Decision Tree Engine with Explicit MISLEADING vs FAKE Boundary
    - Strict 5-Step Probability Calibration & Rebalancing
    - Sentence-Level Claim & Contradiction Extraction
    - Source Credibility & Temporal Awareness
    - 10-Factor Normalized Stylometry
    - Offline Verified Knowledge Grounding
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run model/train.py first.")

        artifact = joblib.load(model_path)
        self.vectorizer = artifact['vectorizer']
        self.classifier = artifact['classifier']
        self.labels = list(self.classifier.classes_)
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def predict(self, text: str) -> dict:
        if not text or not text.strip():
            return {
                'prediction': 'HIGHLY UNCERTAIN',
                'confidence': 0.0,
                'calibrated_confidence': 0.0,
                'real_prob': 0.0,
                'fake_prob': 0.0,
                'misleading_prob': 0.0,
                'label': 'HIGHLY UNCERTAIN',
                'final_label': 'HIGHLY UNCERTAIN',
                'probability_real': 0.0,
                'probability_fake': 0.0,
                'probability_misleading': 0.0,
                'probabilities': {'REAL': 0.0, 'FAKE': 0.0, 'MISLEADING': 0.0},
                'adjusted_probs': {'REAL': 0.0, 'FAKE': 0.0, 'MISLEADING': 0.0},
                'raw_probs': {'REAL': 0.0, 'FAKE': 0.0, 'MISLEADING': 0.0},
                'debunk_strength': 0.0,
                'debunk_score': 0.0,
                'claim_count': 0,
                'correction_count': 0,
                'ratio': 0.0,
                'decision_path': ["Empty input text provided"],
                'reason': 'Empty input text provided.',
                'is_debunking': False,
                'sentence_analysis': {'sentences': [], 'claim_count': 0, 'correction_count': 0},
                'source_credibility': {},
                'knowledge_grounding': None,
                'stylometry': {},
                'misleading_score': 0.0,
                'clean_text': '',
                'top_features': []
            }

        raw_words = text.strip().split()

        # Step 1: Sentence Claim Extraction & Dissection
        sentence_analysis = extract_and_classify_sentences(text)

        # Step 2: Source Credibility & Temporal Awareness
        source_credibility = calculate_source_credibility(text)

        # Step 3: Offline Knowledge Base Match
        knowledge_grounding = match_knowledge_grounding(text)

        # Step 4: 10-Factor Normalized Stylometry
        stylometry = extract_stylometric_features(text)

        # Step 5: Statistical Vectorization & Calibrated Model Prediction
        cleaned = clean_text(text)
        if not cleaned.strip():
            cleaned = text.lower().strip()

        tfidf_vec = self.vectorizer.transform([cleaned])

        raw_real, raw_fake, raw_misleading = 0.33, 0.33, 0.34
        if hasattr(self.classifier, 'predict_proba'):
            probas = self.classifier.predict_proba(tfidf_vec)[0]
            prob_map = {}
            for idx, lbl in enumerate(self.labels):
                prob_map[str(lbl).upper()] = float(probas[idx])
            raw_real = prob_map.get('REAL', 0.0)
            raw_fake = prob_map.get('FAKE', 0.0)
            raw_misleading = prob_map.get('MISLEADING', 0.0)

        total_p = raw_real + raw_fake + raw_misleading
        if total_p > 0:
            raw_prob_dict = {
                "REAL": raw_real / total_p,
                "FAKE": raw_fake / total_p,
                "MISLEADING": raw_misleading / total_p
            }
        else:
            raw_prob_dict = {"REAL": 0.3333, "FAKE": 0.3333, "MISLEADING": 0.3334}

        # Step 6: Hierarchical Deterministic Decision Tree with Calibrated Rebalancing
        decision_result = resolve_hierarchical_decision(
            raw_probs=raw_prob_dict,
            sentence_info=sentence_analysis,
            source_info=source_credibility,
            knowledge_info=knowledge_grounding,
            stylometry_info=stylometry,
            text=text
        )

        prediction = decision_result["final_label"]
        confidence = decision_result["confidence"]
        synced_probs = decision_result["probabilities"]
        prob_real = synced_probs["REAL"]
        prob_fake = synced_probs["FAKE"]
        prob_misleading = synced_probs["MISLEADING"]
        decision_path = decision_result["decision_path"]

        # Feature Importance Extraction
        feature_contributions = []
        try:
            nonzero_indices = tfidf_vec.nonzero()[1]
            coefs = None
            if hasattr(self.classifier, 'coef_'):
                coefs = self.classifier.coef_
            elif hasattr(self.classifier, 'calibrated_classifiers_'):
                coefs_list = [c.estimator.coef_ for c in self.classifier.calibrated_classifiers_ if hasattr(c.estimator, 'coef_')]
                if coefs_list:
                    coefs = np.mean(coefs_list, axis=0)

            if coefs is not None:
                if coefs.ndim > 1:
                    norm_pred = 'REAL' if 'REAL' in prediction else 'MISLEADING' if 'MISLEADING' in prediction else 'FAKE'
                    class_idx = self.labels.index(norm_pred) if norm_pred in self.labels else 0
                    row_coefs = coefs[class_idx]
                else:
                    row_coefs = coefs

                for idx in nonzero_indices:
                    word = self.feature_names[idx]
                    score = float(row_coefs[idx] * tfidf_vec[0, idx])
                    feature_contributions.append((word, score))
                
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        except Exception:
            pass

        reason = decision_path[-1] if decision_path else f"Classification resolved as {prediction} with {confidence:.2%} confidence."

        return {
            'prediction': prediction,
            'final_label': prediction,
            'confidence': confidence,
            'calibrated_confidence': confidence,
            'adjusted_probs': synced_probs,
            'probabilities': synced_probs,
            'adjusted_probabilities': synced_probs,
            'real_prob': prob_real,
            'fake_prob': prob_fake,
            'misleading_prob': prob_misleading,
            'label': prediction,
            'probability_real': round(prob_real * 100, 2),
            'probability_fake': round(prob_fake * 100, 2),
            'probability_misleading': round(prob_misleading * 100, 2),
            'raw_probs': raw_prob_dict,
            'raw_prob_real': round(raw_prob_dict['REAL'], 4),
            'raw_prob_fake': round(raw_prob_dict['FAKE'], 4),
            'raw_prob_misleading': round(raw_prob_dict['MISLEADING'], 4),
            'debunk_strength': 0.85 if 'Debunk' in prediction else 0.0,
            'debunk_score': 0.85 if 'Debunk' in prediction else 0.0,
            'claim_count': sentence_analysis.get('claim_count', 0),
            'correction_count': sentence_analysis.get('correction_count', 0),
            'ratio': sentence_analysis.get('claim_to_correction_ratio', 0.0),
            'misleading_score': decision_result.get('misleading_score', 0.0),
            'decision_path': decision_path,
            'reason': reason,
            'is_debunking': ('Debunk' in prediction),
            'sentence_analysis': sentence_analysis,
            'source_credibility': source_credibility,
            'knowledge_grounding': knowledge_grounding,
            'stylometry': stylometry,
            'clean_text': cleaned,
            'top_features': feature_contributions[:10]
        }
