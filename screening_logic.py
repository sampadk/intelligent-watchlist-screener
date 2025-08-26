# screening_logic.py
# V3 - Batch Onomastic Classification & Complete Rule-Based Fallback

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

def load_data(filepath: str = "sdn.csv"):
    """Loads the sanctions list CSV."""
    df = pd.read_csv(filepath, header=None)
    df.columns = ['ent_num', 'name', 'type', 'program', 'title', 'call_sign', 'vess_type', 'tonnage', 'grt', 'vess_flag', 'vess_owner', 'remarks']
    df['clean_name'] = df['name'].str.lower().str.strip()
    return df

# --- 3. Onomastic Classification Engine (Rules & LLM) ---

# Regex constants for the complete rule-based classifier
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
    """Comprehensive heuristic fallback classifier based on your friend's logic."""
    n = name.strip()
    n_l = n.lower()
    probs: Dict[str, float] = {tid: 0.0 for tid in types_index.ordered_ids}

    # Script-based quick wins
    if re.search(_CHINESE_CHAR_RANGE, n): _vote_prob("han_chinese_pinyin", probs, 0.9)
    elif re.search(_ARABIC_CHAR_RANGE, n): _vote_prob("arabic_urdu_persian_latinized", probs, 0.8)
    elif re.search(_CYRILLIC_CHAR_RANGE, n):
        _vote_prob("russian_east_slavic_patronymic", probs, 0.6)
        _vote_prob("ukrainian_latinized", probs, 0.2)
    elif re.search(_THAI_CHAR_RANGE, n): _vote_prob("thai", probs, 0.85)

    # Particles & patterns
    if re.search(_ARABIC_PARTICLES, n_l): _vote_prob("arabic_urdu_persian_latinized", probs, 0.5)
    if re.search(_DUTCH_PARTICLES, n_l): _vote_prob("dutch_belgian_particles", probs, 0.6)
    if re.search(_IRISH_PREFIXES, n_l): _vote_prob("irish_scottish_gaelic_prefixes", probs, 0.7)
    if "-" in n or " " in n and any(tok for tok in n.split() if "-" in tok): _vote_prob("romance_hyphenated_double_barrel", probs, 0.3)

    # Anglo-like token structure
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
    """OPTIMIZED: Classifies a batch of names in a single LLM call."""
    if not names or not model:
        return {name: classify_rules(name, types_index) for name in names}

    unique_names = sorted(list(set(names)))
    name_list_str = "\n".join([f'- "{name}"' for name in unique_names])
    type_id_list = ", ".join(types_index.ordered_ids)

    prompt = f"""You are an onomastics expert for watchlist screening. For each name in the provided list, classify it into the most likely onomastic type.

Valid type IDs (use EXACTLY these):
{type_id_list}

Names to classify:
{name_list_str}

Respond with a single, compact JSON object. The keys of the object should be the names from the list. Each value should be another JSON object with two keys: "top_id" (the single best type ID) and "probabilities" (a dictionary of type IDs to probability scores, summing to 1.0).

Example response format:
{{
  "John Smith": {{"top_id": "english_anglo", "probabilities": {{"english_anglo": 0.95, "irish_scottish_gaelic_prefixes": 0.05}}}},
  "Wang Wei": {{"top_id": "han_chinese_pinyin", "probabilities": {{"han_chinese_pinyin": 1.0}}}}
}}
"""
    try:
        response = model.generate_content(prompt)
        json_response_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        parsed_batch = json.loads(json_response_str)
        
        results = {}
        for name in unique_names:
            parsed_result = parsed_batch.get(name)
            if parsed_result and "top_id" in parsed_result and "probabilities" in parsed_result:
                top_id = parsed_result["top_id"]
                probs = parsed_result["probabilities"]
                # Normalize probabilities just in case
                total_prob = sum(probs.values()) or 1.0
                normalized_probs = {k: v / total_prob for k, v in probs.items()}
                
                results[name] = NameAnalysisResult(
                    name=name, top_type_id=top_id,
                    type_display_name=types_index.by_id.get(top_id, NameType(id=top_id, display_name=top_id, weights={})).display_name,
                    probabilities=normalized_probs, engine="LLM (Batch)", llm_info=parsed_result
                )
            else:
                results[name] = classify_rules(name, types_index) # Fallback for this specific name if LLM missed it
        return results

    except Exception as e:
        # Fallback to rules for the entire batch if LLM call fails
        return {name: classify_rules(name, types_index) for name in unique_names}

# --- 4. Similarity Scoring Engine ---
def calculate_all_scores(name1: str, name2: str) -> Dict[str, float]:
    """Calculates all four pairwise similarity scores."""
    n1 = (name1 or "").strip().lower()
    n2 = (name2 or "").strip().lower()

    scores = {
        "jaro_winkler": jellyfish.jaro_winkler_similarity(n1, n2),
        "levenshtein_normalized": (1.0 - (jellyfish.levenshtein_distance(n1, n2) / max(len(n1), len(n2)))) if max(len(n1), len(n2)) > 0 else 0.0,
        "thefuzz_token_set_ratio": fuzz.token_set_ratio(n1, n2) / 100.0
    }
    
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    try:
        X = tfidf.fit_transform([n1, n2])
        scores["cosine_similarity"] = float(cosine_similarity(X[0:1], X[1:2])[0][0])
    except ValueError:
        scores["cosine_similarity"] = 0.0

    return {k: round(v, 4) for k, v in scores.items()}

def get_blended_weights(analysis1: NameAnalysisResult, analysis2: NameAnalysisResult, types_index: TypesIndex) -> Dict[str, float]:
    """Averages the weights from two different name type classifications."""
    weights1 = types_index.by_id.get(analysis1.top_type_id, NameType(id="",display_name="",weights=types_index.default_weights)).weights or types_index.default_weights
    weights2 = types_index.by_id.get(analysis2.top_type_id, NameType(id="",display_name="",weights=types_index.default_weights)).weights or types_index.default_weights

    blended_weights = {
        key: (weights1.get(key, 0) + weights2.get(key, 0)) / 2
        for key in types_index.default_weights.keys()
    }
    return blended_weights

def get_weighted_ensemble_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """Calculates the final weighted score after normalizing weights."""
    total_weight = sum(weights.values()) or 1.0
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    final_score = sum(scores.get(k, 0.0) * normalized_weights.get(k, 0.0) for k in weights)
    return round(final_score, 4)