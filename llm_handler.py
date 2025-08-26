import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
from screening_logic import NameAnalysisResult # Import the dataclass for type hinting

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
REASONING_MODEL = genai.GenerativeModel('gemini-1.5-pro-latest')

def get_llm_assessment(input_name: str, matched_name: str, screening_data: dict) -> dict | None:
    
    input_analysis: NameAnalysisResult = screening_data['input_name_analysis']
    candidate_analysis: NameAnalysisResult = screening_data['candidate_name_analysis']
    
    # Get the confidence score for the top type classification
    input_confidence = input_analysis.probabilities.get(input_analysis.top_type_id, 0.0)
    candidate_confidence = candidate_analysis.probabilities.get(candidate_analysis.top_type_id, 0.0)

    prompt = f"""
    You are an expert financial crime compliance analyst. Your task is to provide a detailed, structured risk assessment for a potential sanctions list match based on a sophisticated onomastic analysis.

    **Case Details:**
    - Name to be Screened: "{input_name}"
      - Classified as: '{input_analysis.type_display_name}' (Confidence: {input_confidence:.0%})
    - Potential Match from Watchlist: "{matched_name}"
      - Classified as: '{candidate_analysis.type_display_name}' (Confidence: {candidate_confidence:.0%})

    **Evidence Dossier:**
    1.  **Onomastic Weights Used (Blended):** {json.dumps(screening_data['blended_weights_used'])}
    2.  **Raw Similarity Scores:** {json.dumps(screening_data['raw_scores'])}
    3.  **Final Weighted Ensemble Score:** {screening_data['weighted_score']}

    **Your Task:**
    Based on the complete evidence dossier, generate a comprehensive assessment. Your response MUST be a single, valid JSON object with the following structure:
    {{
      "final_risk_score": <An integer from 0 to 100 representing your final confidence in the match.>,
      "reasoning": "<A detailed paragraph explaining your score. Justify your conclusion by referencing how the onomastic classifications (and their confidence scores) influenced the weights and how the raw scores support or contradict the final weighted score.>",
      "key_parameters": [
        "Input name classified as '{input_analysis.type_display_name}' ({input_confidence:.0%} confidence).",
        "Candidate name classified as '{candidate_analysis.type_display_name}' ({candidate_confidence:.0%} confidence).",
        "The blended weights prioritized the '{max(screening_data['blended_weights_used'], key=screening_data['blended_weights_used'])}' algorithm.",
        "The final data-driven weighted score was {screening_data['weighted_score']}."
      ]
    }}
    """
    
    try:
        response = REASONING_MODEL.generate_content(prompt)
        json_response_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(json_response_str)
        return result
    except Exception as e:
        print(f"An error occurred with the LLM or JSON parsing: {e}")
        return {"final_risk_score": 0, "reasoning": f"Error during analysis: {e}", "key_parameters": []}