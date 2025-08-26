# llm_handler.py
# V5.1 - Fixed JSON serialization error for NameAnalysisResult object

import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from typing import List, Dict, Any
from screening_logic import NameAnalysisResult # Import for type hinting

load_dotenv()

# Initialize the model once
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    REASONING_MODEL = genai.GenerativeModel('models/gemini-2.5-flash-lite')
except (TypeError, ValueError) as e:
    REASONING_MODEL = None
    print(f"Error initializing Google Generative AI: {e}")

def _make_data_serializable(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Helper function to convert nested NameAnalysisResult objects into dictionaries.
    """
    serializable_data = []
    for item in data:
        new_item = item.copy()
        # Convert the dataclass objects to simple dictionaries
        if 'input_analysis' in new_item and isinstance(new_item['input_analysis'], NameAnalysisResult):
            new_item['input_analysis'] = new_item['input_analysis'].__dict__
        if 'candidate_analysis' in new_item and isinstance(new_item['candidate_analysis'], NameAnalysisResult):
            new_item['candidate_analysis'] = new_item['candidate_analysis'].__dict__
        serializable_data.append(new_item)
    return serializable_data

def get_batch_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top INDIVIDUAL candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    # THE FIX: Convert custom objects to dictionaries before serializing
    serializable_candidates = _make_data_serializable(candidates_data)

    prompt = f"""You are a financial crime compliance analyst. For each candidate in the JSON list below, provide a risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{json.dumps(serializable_candidates, indent=2)}"""
    
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

    # This function already sends simple data, so no changes are needed here, but it's good practice.
    prompt = f"""You are a compliance analyst. For each non-individual entity in the JSON list below, provide a brief risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{json.dumps(candidates_data, indent=2)}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch entity LLM: {e}")
        return {item['candidate_name']: {"final_risk_score": 0, "reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}
