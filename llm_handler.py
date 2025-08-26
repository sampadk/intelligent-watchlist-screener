# llm_handler.py
# V5.4 - Reworked prompt to focus on qualitative match reasoning instead of risk scoring.

import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from dataclasses import is_dataclass, asdict
from screening_logic import NameAnalysisResult # Import for type hinting

load_dotenv()

# Initialize the model once
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    REASONING_MODEL = genai.GenerativeModel('models/gemini-2.5-pro-latest')
except (TypeError, ValueError) as e:
    REASONING_MODEL = None
    print(f"Error initializing Google Generative AI: {e}")

# --- A robust JSON encoder for our custom dataclass objects ---
class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def get_batch_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top INDIVIDUAL candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    try:
        prompt_payload = json.dumps(candidates_data, indent=2, cls=DataclassJSONEncoder)
    except Exception as e:
        print(f"Error serializing data for LLM prompt: {e}")
        return {item['candidate_name']: {"reasoning": f"Error serializing data: {e}", "key_parameters": []} for item in candidates_data}

    prompt = f"""You are a financial crime compliance analyst. Your task is to provide a qualitative analysis explaining why two names are a potential match.

**Analysis Instructions:**
1.  **For very high scores (`weighted_score` > 0.95):** The names are a likely exact match. Your reasoning should simply confirm the high degree of similarity.
2.  **For close scores (`weighted_score` > 0.7):** Analyze the `raw_scores` to explain the match. Your reasoning MUST suggest potential causes like:
    * **Typo / Fat-finger error:** (e.g., high Jaro-Winkler, but slightly lower Levenshtein).
    * **Initials vs. Full Name:** (e.g., high TheFuzz Token Set Ratio).
    * **Alternate Transliteration/Spelling:** (e.g., high Cosine Similarity and TheFuzz).
    * **Reordered Name Tokens:** (e.g., very high TheFuzz Token Set Ratio).
    * Do not limit yourself to these examples; interpret the scores to provide a plausible explanation.
3.  **Do NOT invent a risk score.** Your entire output is the explanation.

Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'reasoning' and 'key_parameters'.

{prompt_payload}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch individual LLM: {e}")
        return {item['candidate_name']: {"reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}

def get_batch_entity_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top ENTITY candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    prompt = f"""You are a compliance analyst. For each non-individual entity in the JSON list below, provide a brief explanation for why the names are a potential match, focusing on the normalized names and the match score. Do not invent a risk score.

Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'reasoning' and 'key_parameters'.

{json.dumps(candidates_data, indent=2, cls=DataclassJSONEncoder)}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch entity LLM: {e}")
        return {item['candidate_name']: {"reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}
