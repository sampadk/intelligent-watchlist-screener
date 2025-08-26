import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from screening_logic import (
    load_data,
    load_name_type_weights,
    calculate_all_scores,
    get_blended_weights,
    get_weighted_ensemble_score,
    batch_analyze_names_with_llm  # The new batch function
)
from llm_handler import get_llm_assessment

st.set_page_config(layout="wide")
st.title("🤖 Onomastic-Aware Watchlist Screener")
st.caption("An advanced screener using single-shot LLM name classification and blended weighting.")

# --- Initialization ---
@st.cache_resource
def init_models_and_data():
    """Load all data and initialize models once."""
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found. Please set it in your secrets.")
        return None, None, None
    
    genai.configure(api_key=api_key)
    classification_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    
    types_index = load_name_type_weights("name_types.json")
    sdn_df = load_data("sdn.csv")
    return types_index, sdn_df, classification_model

types_index, sdn_df, classification_model = init_models_and_data()

if sdn_df is None:
    st.stop()

# --- UI Elements ---
input_name = st.text_input("Enter the full name of the individual or entity to screen:", "Abdelaziz Bouteflika")
screen_button = st.button("Run Advanced Screening")

# --- Logic on Button Click ---
if screen_button and input_name:
    # 1. Pre-filter to get a manageable list of candidates
    candidates_df = sdn_df[sdn_df['clean_name'].str.contains(input_name.split()[0].lower(), na=False)]
    if len(candidates_df) > 50:
        candidates_df = candidates_df.head(50)

    if candidates_df.empty:
        st.success("✅ No potential matches found after initial filtering.")
    else:
        st.info(f"Found {len(candidates_df)} candidates. Performing efficient batch analysis...")
        
        # 2. BATCH CLASSIFICATION - The core optimization
        candidate_names = candidates_df['name'].tolist()
        names_to_classify = [input_name] + candidate_names
        with st.spinner(f"Classifying {len(set(names_to_classify))} unique names in a single LLM call..."):
            analysis_results = batch_analyze_names_with_llm(names_to_classify, classification_model, types_index)

        progress_bar = st.progress(0, text="Scoring candidates...")
        results = []
        
        # 3. Loop through candidates to score them (NO LLM calls inside this loop)
        for i, row in enumerate(candidates_df.itertuples()):
            candidate_name = row.name
            
            input_analysis = analysis_results[input_name]
            candidate_analysis = analysis_results[candidate_name]
            
            raw_scores = calculate_all_scores(input_name, candidate_name)
            blended_weights = get_blended_weights(input_analysis, candidate_analysis, types_index)
            weighted_score = get_weighted_ensemble_score(raw_scores, blended_weights)
            
            screening_data = {
                "input_name_analysis": input_analysis,
                "candidate_name_analysis": candidate_analysis,
                "raw_scores": raw_scores,
                "blended_weights_used": blended_weights,
                "weighted_score": weighted_score,
            }
            
            # 4. Pass the full dossier to the SECOND (reasoning) LLM
            llm_assessment = get_llm_assessment(input_name, candidate_name, screening_data)
            
            if llm_assessment:
                results.append({
                    "candidate_name": candidate_name,
                    "screening_data": screening_data,
                    "llm_assessment": llm_assessment,
                    "full_record": sdn_df.loc[row.Index].to_dict()
                })

            progress_bar.progress((i + 1) / len(candidates_df), text=f"Scoring candidate {i+1}/{len(candidates_df)}")
        
        progress_bar.empty()
        st.subheader("Screening Results")
        
        # (The rest of the display logic is identical to the previous version)
        sorted_results = sorted(results, key=lambda x: x['llm_assessment']['final_risk_score'], reverse=True)

        for result in sorted_results:
            score = result['llm_assessment']['final_risk_score']
            color = "red" if score > 85 else "orange" if score > 70 else "blue"
            
            with st.container(border=True):
                # Display logic remains the same...
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"#### Match: **{result['candidate_name']}**")
                with col2:
                    st.markdown(f"<h4 style='text-align: right; color:{color};'>Final Risk Score: {score}</h4>", unsafe_allow_html=True)

                st.markdown("**LLM Reasoning:**")
                st.info(result['llm_assessment']['reasoning'])
                
                with st.expander("Show Full Analysis Dossier"):
                    input_analysis = result['screening_data']['input_name_analysis']
                    candidate_analysis = result['screening_data']['candidate_name_analysis']
                    
                    st.markdown("**Onomastic Classification**")
                    ana_col1, ana_col2 = st.columns(2)
                    with ana_col1:
                        st.write(f"**Input Name:** `{input_analysis.name}`")
                        st.write(f"**Type:** `{input_analysis.type_display_name}` (Confidence: {input_analysis.probabilities.get(input_analysis.top_type_id, 0.0):.0%})")
                        st.write(f"**Engine:** `{input_analysis.engine}`")
                    with ana_col2:
                        st.write(f"**Candidate Name:** `{candidate_analysis.name}`")
                        st.write(f"**Type:** `{candidate_analysis.type_display_name}` (Confidence: {candidate_analysis.probabilities.get(candidate_analysis.top_type_id, 0.0):.0%})")
                        st.write(f"**Engine:** `{candidate_analysis.engine}`")
                    
                    st.markdown("**Evidence & Scoring**")
                    ev_col1, ev_col2, ev_col3 = st.columns(3)
                    with ev_col1:
                        st.write("**Blended Weights Used:**")
                        st.json(result['screening_data']['blended_weights_used'], expanded=False)
                    with ev_col2:
                        st.write("**Raw Similarity Scores:**")
                        st.json(result['screening_data']['raw_scores'], expanded=False)
                    with ev_col3:
                        st.write("**Weighted Ensemble Score:**")
                        st.progress(result['screening_data']['weighted_score'], text=f"{result['screening_data']['weighted_score']:.4f}")
                    
                    st.markdown("**Key Parameters from LLM:**")
                    for param in result['llm_assessment']['key_parameters']:
                        st.markdown(f"- *{param}*")
                        
                    st.markdown("**Full Watchlist Record**")
                    st.json(result['full_record'], expanded=False)