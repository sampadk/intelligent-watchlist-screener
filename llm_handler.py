# llm_handler.py
# V5.5 - Fixed KeyError in exception block and updated model name.

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
    # UPDATED to use the correct model name
    REASONING_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-lite')
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
        return {item.get('candidate_name', 'Unknown'): {"reasoning": f"Error serializing data: {e}", "key_parameters": []} for item in candidates_data}

    prompt = f"""You are a financial crime compliance analyst. For each candidate in the JSON list below, provide a risk assessment.

**Crucial Instruction:** If a candidate's `weighted_score` is very high (e.g., > 0.95), the names are a likely exact match. In this case, your reasoning MUST focus on confirming the high degree of similarity and downplay or ignore minor onomastic classification differences. Do not express surprise that identical names were classified differently; this is expected noise. Instead, confirm the clear match.

Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{prompt_payload}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch individual LLM: {e}")
        # THE FIX: Correctly access the candidate_name from the original data structure.
        return {item.get('screening_data', {}).get('candidate_name', 'Unknown'): {"reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}

def get_batch_entity_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top ENTITY candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    prompt = f"""You are a compliance analyst. For each non-individual entity in the JSON list below, provide a brief risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{json.dumps(candidates_data, indent=2, cls=DataclassJSONEncoder)}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch entity LLM: {e}")
        # THE FIX: Correctly access the candidate_name from the original data structure.
        return {item.get('candidate_name', 'Unknown'): {"reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}
