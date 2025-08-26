# app.py
# V5.1 - FINAL VERSION with corrected display logic for all entity types

import streamlit as st
import pandas as pd
import os
import google.generativeai as genai
from screening_logic import *
from llm_handler import get_batch_llm_assessment, get_batch_entity_llm_assessment

st.set_page_config(layout="wide", page_title="Advanced Watchlist Screener")
st.title("🤖 Onomastic-Aware & Entity-Smart Watchlist Screener")

# --- Initialization ---
@st.cache_resource
def init_models_and_data():
    """Load all data and initialize models once."""
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    if not api_key:
        return None, None, None
    
    genai.configure(api_key=api_key)
    classification_model = genai.GenerativeModel('models/gemini-2.5-flash-lite')
    types_index = load_name_type_weights("name_types.json")
    sdn_df = load_data("sdn_classified.parquet")
    return types_index, sdn_df, classification_model

types_index, sdn_df, classification_model = init_models_and_data()

if sdn_df is None:
    st.error("Fatal Error: Could not load `sdn_classified.parquet`. Please run `preprocess_sdn.py` first.")
    st.stop()

# --- UI Controls in Sidebar ---
with st.sidebar:
    st.header("⚙️ Screening Controls")
    entity_type = st.selectbox("Select Entity Type to Screen", ('Individual', 'Entity', 'Vessel', 'Aircraft'))
    score_threshold = st.slider("Match Score Threshold", min_value=0.5, max_value=1.0, value=0.7, step=0.01)
    page_size = st.slider("Matches to retrieve per batch", min_value=1, max_value=10, value=5)

input_name = st.text_input("Enter the full name to screen:", "KIM Jong Un")

if st.button("Run Advanced Screening"):
    st.session_state.clear()
    if not input_name:
        st.warning("Please enter a name.")
    else:
        all_results = []
        if entity_type == 'Individual':
            with st.spinner("Analyzing input name and performing smart filtering..."):
                input_analysis = batch_analyze_names_with_llm([input_name], classification_model, types_index)[input_name]
                individuals_df = sdn_df[sdn_df['type'] == 'individual'].copy()
                top_candidates_df = get_top_candidates_smart_filter(input_analysis, individuals_df, limit=50)
            
            with st.spinner("Calculating final scores..."):
                for row in top_candidates_df.itertuples():
                    candidate_analysis = NameAnalysisResult(name=row.name, top_type_id=row.top_type_id, type_display_name=row.type_display_name, probabilities=row.probabilities if pd.notna(row.probabilities) else {}, engine=row.engine)
                    raw_scores = calculate_all_scores(input_name, row.name)
                    blended_weights = get_blended_weights(input_analysis, candidate_analysis, types_index)
                    weighted_score = get_weighted_ensemble_score(raw_scores, blended_weights)
                    
                    if weighted_score >= score_threshold:
                        all_results.append({"candidate_name": row.name, "final_score": weighted_score, "screening_data": {"input_name_analysis": input_analysis, "candidate_analysis": candidate_analysis, "raw_scores": raw_scores, "blended_weights_used": blended_weights, "weighted_score": weighted_score}, "full_record": sdn_df.loc[row.Index].to_dict()})
        else: # Logic for Entity, Vessel, Aircraft
            with st.spinner(f"Screening for non-individual entities..."):
                entity_map = {'Entity': 'entity', 'Vessel': 'vessel', 'Aircraft': 'aircraft'}
                target_df = sdn_df[sdn_df['type'] == entity_map.get(entity_type)].copy()
                
                for row in target_df.itertuples():
                    match_details = normalize_and_match_entity(input_name, row.name)
                    if match_details['match_score'] >= score_threshold:
                        all_results.append({"candidate_name": row.name, "final_score": match_details['match_score'], "screening_data": {"input_name": input_name, "candidate_name": row.name, "match_details": match_details}, "full_record": sdn_df.loc[row.Index].to_dict()})

        st.session_state.results = sorted(all_results, key=lambda x: x['final_score'], reverse=True)
        st.session_state.total_matches = len(st.session_state.results)
        st.success(f"Analysis complete. Found {st.session_state.total_matches} potential matches above the {score_threshold} threshold.")

# --- Display Logic ---
if 'results' in st.session_state and st.session_state.results:
    page = st.session_state.get('page_number', 0)
    start_idx = page * page_size
    end_idx = start_idx + page_size
    results_to_display = st.session_state.results[start_idx:end_idx]
    
    if results_to_display:
        st.subheader(f"Displaying matches {start_idx + 1} to {min(end_idx, st.session_state.total_matches)} of {st.session_state.total_matches}")
        
        with st.spinner("Getting detailed explanations from reasoning LLM..."):
            if entity_type == 'Individual':
                llm_assessments = get_batch_llm_assessment([res['screening_data'] for res in results_to_display])
            else:
                llm_assessments = get_batch_entity_llm_assessment([res['screening_data'] for res in results_to_display])

        if not llm_assessments:
            st.error("Could not get explanations from the reasoning LLM.")
        else:
            for result in results_to_display:
                assessment = llm_assessments.get(result['candidate_name'])
                if not assessment: continue
                
                score = assessment['final_risk_score']
                color = "red" if score > 85 else "orange" if score > 70 else "blue"

                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1: st.markdown(f"#### Match: **{result['candidate_name']}**")
                    with col2: st.markdown(f"<h4 style='text-align: right; color:{color};'>Final Risk Score: {score}</h4>", unsafe_allow_html=True)

                    st.markdown("**LLM Reasoning:**")
                    st.info(assessment['reasoning'])
                    
                    with st.expander("Show Full Analysis Dossier"):
                        # --- CORRECTED DISPLAY LOGIC ---
                        if entity_type == 'Individual':
                            sd = result['screening_data']
                            st.markdown("##### Onomastic Classification")
                            c1, c2 = st.columns(2)
                            c1.metric("Input Name Type", sd['input_analysis'].type_display_name, f"{sd['input_analysis'].engine} @ {sd['input_analysis'].probabilities.get(sd['input_analysis'].top_type_id, 0.0):.0%} conf.")
                            c2.metric("Candidate Name Type", sd['candidate_analysis'].type_display_name, f"{sd['candidate_analysis'].engine} @ {sd['candidate_analysis'].probabilities.get(sd['candidate_analysis'].top_type_id, 0.0):.0%} conf.")
                            st.markdown("##### Evidence & Scoring")
                            c1, c2, c3 = st.columns(3)
                            c1.write("**Blended Weights Used:**"); c1.json(sd['blended_weights_used'], expanded=False)
                            c2.write("**Raw Similarity Scores:**"); c2.json(sd['raw_scores'], expanded=False)
                            c3.metric("Weighted Score", f"{sd['weighted_score']:.4f}")
                        else: # Entity Display Logic
                            sd = result['screening_data']['match_details']
                            st.markdown("##### Entity Normalization & Matching")
                            c1, c2, c3 = st.columns(3)
                            c1.text_area("Input", value=f"Original: {sd['original_1']}\n---\nNormalized: {sd['normalized_1']}", height=120)
                            c2.text_area("Candidate", value=f"Original: {sd['original_2']}\n---\nNormalized: {sd['normalized_2']}", height=120)
                            c3.metric("Normalized Match Score", f"{sd['match_score']:.4f}")

                        st.markdown("##### Key Parameters from LLM")
                        for param in assessment['key_parameters']: st.markdown(f"- *{param}*")
                        st.markdown("##### Full Watchlist Record"); st.json(result['full_record'], expanded=False)

    if end_idx < st.session_state.total_matches:
        if st.button("Retrieve Additional Matches"):
            st.session_state.page_number = st.session_state.get('page_number', 0) + 1
            st.rerun()
    elif st.session_state.total_matches > 0:
        st.info("No more matches to display.")
