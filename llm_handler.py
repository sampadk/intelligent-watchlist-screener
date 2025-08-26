# llm_handler.py
# V5 - Contains dedicated batch reasoners for individuals and entities

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
    REASONING_MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')
except (TypeError, ValueError) as e:
    REASONING_MODEL = None
    print(f"Error initializing Google Generative AI: {e}")

def get_batch_llm_assessment(candidates_data: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Sends a batch of top INDIVIDUAL candidates to the LLM for reasoning in a single call."""
    if not candidates_data or not REASONING_MODEL:
        return None

    prompt = f"""You are a financial crime compliance analyst. For each candidate in the JSON list below, provide a risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{json.dumps(candidates_data, indent=2)}"""
    
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

    prompt = f"""You are a compliance analyst. For each non-individual entity in the JSON list below, provide a brief risk assessment. Your response MUST be a single JSON object where keys are the candidate names and values are another JSON object with 'final_risk_score', 'reasoning', and 'key_parameters'.

{json.dumps(candidates_data, indent=2)}"""
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        return json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    except Exception as e:
        print(f"An error occurred with the batch entity LLM: {e}")
        return {item['candidate_name']: {"final_risk_score": 0, "reasoning": f"Error: {e}", "key_parameters": []} for item in candidates_data}
