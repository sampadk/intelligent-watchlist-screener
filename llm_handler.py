import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-pro-latest')

def get_llm_assessment(input_name, matched_name, scores):
    prompt = f"""
    You are an expert financial crime compliance analyst. Your task is to assess the similarity between two names based on a full ensemble of calculated scores and provide a clear, concise risk assessment.

    **Input Data:**
    - Name to be Screened: "{input_name}"
    - Potential Match from Watchlist: "{matched_name}"
    - Similarity Scores (all scores are 0 to 1, higher is a closer match):
      - Jaro-Winkler Score: {scores['jaro_winkler']} (Sensitive to matching characters at the beginning of the string)
      - Normalized Levenshtein Score: {scores['levenshtein_normalized']} (Based on the number of edits to make strings identical)
      - TheFuzz Token Set Ratio: {scores['thefuzz_token_set_ratio']} (Excellent for handling different word orders and partial matches)
      - Cosine Similarity Score: {scores['cosine_similarity']} (Measures similarity based on shared character patterns)
      
    **Your Task:**
    Based *only* on the provided names and the full ensemble of four scores, perform the following two actions:
    1.  **Generate a "Combined Risk Score"**: An integer from 0 to 100 representing your confidence in the match. A score above 85 should be considered high risk.
    2.  **Write a "Summary Explanation"**: A brief, one or two-sentence explanation for your score. Mention which factors (e.g., strong Jaro-Winkler and TheFuzz scores) influenced your decision.

    **Output Format:**
    Return your response as a valid JSON object with two keys: "combined_risk_score" and "summary_explanation". Do not include any other text or formatting.
    """
    
    try:
        response = model.generate_content(prompt)
        # Clean up the response to ensure it's valid JSON
        json_response_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        result = json.loads(json_response_str)
        return result
    except Exception as e:
        print(f"An error occurred with the LLM: {e}")
        return None