# screening_logic.py
# V5.2 - Final version with all logic for onomastics and robust entity matching.

import json
import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

import pandas as pd
import jellyfish
from thefuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# --- 1. Data Structures ---
@dataclass
class NameType:
    id: str
    display_name: str
    weights: Dict[str, float]

@dataclass
class NameAnalysisResult:
    name: str
    top_type_id: str
    type_display_name: str
    probabilities: Dict[str, float]
    engine: str = "Rules"
    llm_info: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TypesIndex:
    by_id: Dict[str, NameType]
    default_weights: Dict[str, float]
    ordered_ids: List[str]

# --- 2. Loaders & Configuration ---
def load_name_type_weights(filepath: str = "name_types.json") -> TypesIndex:
    """Loads the onomastics JSON file and indexes it by ID."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    types = {}
    ordered_ids = [t["id"] for t in data.get("types", [])]

    for t in data.get("types", []):
        types[t["id"]] = NameType(
            id=t["id"],
            display_name=t.get("display_name", t["id"]),
            weights=t.get("weights", {})
        )
    
    return TypesIndex(
        by_id=types,
        default_weights=data.get("default_weights", {}),
        ordered_ids=ordered_ids
    )

def load_data(filepath: str = "sdn_classified.parquet"):
    """Loads the pre-processed and classified sanctions list."""
    try:
        return pd.read_parquet(filepath)
    except FileNotFoundError:
        return None # The main app will handle this user-facing error

# --- 3. Onomastic Classification Engine (Rules & LLM) ---
_ARABIC_PARTICLES = r"\b(al|el|bin|ibn|bint|abu|abd|al-)\b"
_DUTCH_PARTICLES = r"\b(van|de|der|van der|van de)\b"
_IRISH_PREFIXES = r"^(o'|ó\s|mc|mac|nic)"
_CHINESE_CHAR_RANGE = r"[\u4e00-\u9fff]"
_ARABIC_CHAR_RANGE = r"[\u0600-\u06ff]"
_CYRILLIC_CHAR_RANGE = r"[\u0400-\u04FF]"
_THAI_CHAR_RANGE = r"[\u0E00-\u0E7F]"

def _vote_prob(type_id: str, probs: Dict[str, float], amt: float) -> None:
    if type_id in probs:
        probs[type_id] += amt

def classify_rules(name: str, types_index: TypesIndex) -> NameAnalysisResult:
    """Comprehensive heuristic fallback classifier."""
    n, n_l = name.strip(), name.strip().lower()
    probs: Dict[str, float] = {tid: 0.0 for tid in types_index.ordered_ids}

    if re.search(_CHINESE_CHAR_RANGE, n): _vote_prob("han_chinese_pinyin", probs, 0.9)
    elif re.search(_ARABIC_CHAR_RANGE, n): _vote_prob("arabic_urdu_persian_latinized", probs, 0.8)
    elif re.search(_CYRILLIC_CHAR_RANGE, n):
        _vote_prob("russian_east_slavic_patronymic", probs, 0.6)
        _vote_prob("ukrainian_latinized", probs, 0.2)
    elif re.search(_THAI_CHAR_RANGE, n): _vote_prob("thai", probs, 0.85)

    if re.search(_ARABIC_PARTICLES, n_l): _vote_prob("arabic_urdu_persian_latinized", probs, 0.5)
    if re.search(_DUTCH_PARTICLES, n_l): _vote_prob("dutch_belgian_particles", probs, 0.6)
    if re.search(_IRISH_PREFIXES, n_l): _vote_prob("irish_scottish_gaelic_prefixes", probs, 0.7)
    if "-" in n or " " in n and any(tok for tok in n.split() if "-" in tok): _vote_prob("romance_hyphenated_double_barrel", probs, 0.3)

    tokens = re.findall(r"[A-Za-z']+", n)
    if tokens and all(2 <= len(t) <= 12 for t in tokens): _vote_prob("english_anglo", probs, 0.25)

    if sum(probs.values()) < 0.01:
        seed = {"han_chinese_pinyin": 0.25, "indian_subcontinent_latinized": 0.25, "arabic_urdu_persian_latinized": 0.15, "spanish_double_surname": 0.15, "english_anglo": 0.20}
        for k, v in seed.items():
            if k in probs: probs[k] = v
    
    total = sum(probs.values()) or 1.0
    final_probs = {k: v / total for k, v in probs.items()}
    top_id = max(final_probs, key=final_probs.get)
    
    return NameAnalysisResult(
        name=name, top_type_id=top_id,
        type_display_name=types_index.by_id.get(top_id, NameType(id=top_id, display_name=top_id, weights={})).display_name,
        probabilities=final_probs, engine="Rules"
    )

def batch_analyze_names_with_llm(names: List[str], model: genai.GenerativeModel, types_index: TypesIndex) -> Dict[str, NameAnalysisResult]:
    """Classifies a batch of names in a single LLM call."""
    if not names or not model:
        return {name: classify_rules(name, types_index) for name in names}
    unique_names = sorted(list(set(names)))
    name_list_str = "\n".join([f'- "{name}"' for name in unique_names])
    type_id_list = ", ".join(types_index.ordered_ids)
    prompt = f"""You are an onomastics expert. For each name in the list, classify it into the most likely onomastic type.
Valid type IDs: {type_id_list}
Names to classify:
{name_list_str}
Respond with a single, compact JSON object. The keys are the names. Each value should be another JSON object with "top_id" (the single best type ID) and "probabilities" (a dictionary of type IDs to probability scores, summing to 1.0).
Example: {{"John Smith": {{"top_id": "english_anglo", "probabilities": {{"english_anglo": 0.95, "irish_scottish_gaelic_prefixes": 0.05}}}}}}"""
    try:
        response = model.generate_content(prompt)
        parsed_batch = json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
        results = {}
        for name in unique_names:
            parsed = parsed_batch.get(name)
            if parsed and "top_id" in parsed and "probabilities" in parsed:
                total = sum(parsed["probabilities"].values()) or 1.0
                results[name] = NameAnalysisResult(
                    name=name, top_type_id=parsed["top_id"],
                    type_display_name=types_index.by_id.get(parsed["top_id"], NameType(id=parsed["top_id"], display_name=parsed["top_id"], weights={})).display_name,
                    probabilities={k: v / total for k, v in parsed["probabilities"].items()}, engine="LLM (Batch)", llm_info=parsed
                )
            else:
                results[name] = classify_rules(name, types_index)
        return results
    except Exception:
        return {name: classify_rules(name, types_index) for name in unique_names}

# --- 4. Similarity Scoring & Filtering ---
def calculate_all_scores(name1: str, name2: str) -> Dict[str, float]:
    """Calculates all four pairwise similarity scores for individuals."""
    n1, n2 = str(name1 or "").strip().lower(), str(name2 or "").strip().lower()
    scores = {
        "jaro_winkler": jellyfish.jaro_winkler_similarity(n1, n2),
        "levenshtein_normalized": (1.0 - (jellyfish.levenshtein_distance(n1, n2) / max(len(n1), len(n2)))) if max(len(n1), len(n2)) > 0 else 0.0,
        "thefuzz_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0
    }
    try:
        tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit_transform([n1, n2])
        scores["cosine_similarity"] = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        scores["cosine_similarity"] = 0.0
    return {k: round(v, 4) for k, v in scores.items()}

def get_blended_weights(analysis1: NameAnalysisResult, analysis2: NameAnalysisResult, types_index: TypesIndex) -> Dict[str, float]:
    """Averages the weights from two different name type classifications."""
    weights1 = types_index.by_id.get(analysis1.top_type_id, NameType(id="",display_name="",weights=types_index.default_weights)).weights or types_index.default_weights
    weights2 = types_index.by_id.get(analysis2.top_type_id, NameType(id="",display_name="",weights=types_index.default_weights)).weights or types_index.default_weights
    return {key: (weights1.get(key, 0) + weights2.get(key, 0)) / 2 for key in types_index.default_weights.keys()}

def get_weighted_ensemble_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculates the final weighted score after normalizing weights."""
    total_weight = sum(weights.values()) or 1.0
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    return round(sum(scores.get(k, 0.0) * normalized_weights.get(k, 0.0) for k in weights), 4)

def normalize_and_match_entity(name1: str, name2: str) -> dict:
    """Normalizes and matches non-individual entities."""
    n1_norm, n2_norm = str(name1 or "").strip().lower(), str(name2 or "").strip().lower()
    rules = {
        r'\b(limited|ltd)\b': 'ltd', r'\b(company|co)\b': 'co', r'\b(incorporated|inc)\b': 'inc',
        r'\b(corporation|corp)\b': 'corp', r'\b(public limited company|plc)\b': 'plc',
        r'\b(shipping|shpg)\b': 'shp', r'\b(vessel|vsl)\b': '', r'[.,]': ''
    }
    for p, r in rules.items():
        n1_norm, n2_norm = re.sub(p, r, n1_norm), re.sub(p, r, n2_norm)
    
    n1_norm, n2_norm = ' '.join(n1_norm.split()), ' '.join(n2_norm.split())
    
    return {
        "original_1": name1, "original_2": name2, "normalized_1": n1_norm, "normalized_2": n2_norm,
        "match_score": round(fuzz.token_sort_ratio(n1_norm, n2_norm) / 100.0, 4)
    }

def get_top_candidates_smart_filter(input_analysis: NameAnalysisResult, classified_df: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    """Gets top candidates using a smart score combining string and onomastic similarity."""
    input_name, input_type = input_analysis.name.lower(), input_analysis.top_type_id
    def calculate_smart_score(row):
        fuzzy_score = fuzz.partial_ratio(input_name, str(row['name']).lower()) / 100.0
        onomastic_bonus = 0.15 if pd.notna(row['top_type_id']) and row['top_type_id'] == input_type else 0.0
        return fuzzy_score + onomastic_bonus
    classified_df['smart_score'] = classified_df.apply(calculate_smart_score, axis=1)
    return classified_df.nlargest(limit, 'smart_score')
