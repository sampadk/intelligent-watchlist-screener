# llm_handler.py
# V5.2 - Replaced manual serialization with a robust custom JSON Encoder

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
    REASONING_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-lite')
except (TypeError, ValueError) as e:
    REASONING_MODEL = None
    print(f"Error initializing Google Generative AI: {e}")

# --- THE FIX: A robust JSON encoder for our custom dataclass objects ---
class DataclassJSONEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that knows how to convert dataclass objects to dictionaries.
    """
    def default(self, o):
        if is_dataclass(o):
            return asdict(o)
        return super().default(o)

def get_batch_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top INDIVIDUAL candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    # Use the custom encoder to handle the NameAnalysisResult objects
    try:
        # The 'cls' argument tells json.dumps to use our custom encoder
        prompt_payload = json.dumps(candidates_data, indent=2, cls=DataclassJSONEncoder)
    except Exception as e:
        print(f"Error serializing data for LLM prompt: {e}")
        return {item['candidate_name']: {"final_risk_score": 0, "reasoning": f"Error serializing data: {e}", "key_parameters": []} for item in candidates_data}


    prompt = f"""You are a financial crime compliance analyst. For each candidate in the JSON list below, provide a risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{prompt_payload}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch individual LLM: {e}")
        return {item['candidate_name']: {"final_risk_score": 0, "reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}

def get_batch_entity_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top ENTITY candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    # This function already sends simple data, but we use the encoder for consistency.
    prompt = f"""You are a compliance analyst. For each non-individual entity in the JSON list below, provide a brief risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{json.dumps(candidates_data, indent=2, cls=DataclassJSONEncoder)}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch entity LLM: {e}")
        return {item['candidate_name']: {"final_risk_score": 0, "reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}
