# app.py
# V5.9.3 (Debug Version) - Added a debug line and improved 'no matches' messaging.

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

# --- Callback Functions ---
def reset_search():
    """Clears the search results and resets the page."""
    for key in ['results', 'run_search', 'total_matches', 'page_number']:
        if key in st.session_state:
            del st.session_state[key]

# --- UI Controls in Sidebar ---
with st.sidebar:
    st.header("⚙️ Screening Controls")
    
    available_lists = ['OFAC Consolidated']
    selected_lists = st.multiselect(
        "Select Sanctions Lists",
        options=available_lists,
        default=available_lists,
        disabled=True,
        help="Currently locked to the OFAC list. More lists will be added in future iterations."
    )
    
    entity_type = st.selectbox("Select Entity Type", ('Individual', 'Entity', 'Vessel', 'Aircraft'))
    score_threshold = st.slider("Match Score Threshold", 0.5, 1.0, 0.7, 0.01, help="Only show matches with a final data-driven score above this value.")
    page_size = st.slider("Matches per batch", 1, 10, 5)

# --- Main Search Area ---
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    input_name = st.text_input("Enter the full name to screen:", "ZAGROS PETROCHEMICAL", label_visibility="collapsed")
with col2:
    if st.button("Screen Name", type="primary", use_container_width=True):
        st.session_state.run_search = True
with col3:
    st.button("Reset", on_click=reset_search, use_container_width=True)

# Decouple search logic to only run when the button is clicked
if st.session_state.get('run_search', False):
    st.session_state.run_search = False
    st.session_state.results = []
    st.session_state.page_number = 0
    
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
                    probs = row.probabilities if pd.notna(row.probabilities) else {}
                    candidate_analysis = NameAnalysisResult(name=row.name, top_type_id=row.top_type_id, type_display_name=row.type_display_name, probabilities=probs, engine=row.engine)
                    raw_scores = calculate_all_scores(input_name, row.name)
                    blended_weights = get_blended_weights(input_analysis, candidate_analysis, types_index)
                    weighted_score = get_weighted_ensemble_score(raw_scores, blended_weights)
                    
                    if weighted_score >= score_threshold:
                        all_results.append({"candidate_name": row.name, "final_score": weighted_score, "screening_data": {"input_analysis": input_analysis, "candidate_analysis": candidate_analysis, "raw_scores": raw_scores, "blended_weights_used": blended_weights, "weighted_score": weighted_score}, "full_record": sdn_df.loc[row.Index].to_dict()})
        else: # Logic for Entity, Vessel, Aircraft
            with st.spinner(f"Screening for non-individual entities..."):
                entity_map = {'Entity': 'entity', 'Vessel': 'vessel', 'Aircraft': 'aircraft'}
                target_df = sdn_df[sdn_df['type'] == entity_map.get(entity_type)].copy()
                
                # --- DEBUGGING LINE ---
                st.info(f"DEBUG: Found {len(target_df)} records of type '{entity_type}' in the data file before matching.")
                
                for row in target_df.itertuples():
                    match_details = normalize_and_match_entity(input_name, row.name)
                    if match_details['match_score'] >= score_threshold:
                        all_results.append({"candidate_name": row.name, "final_score": match_details['match_score'], "screening_data": {"input_name": input_name, "candidate_name": row.name, "match_details": match_details}, "full_record": sdn_df.loc[row.Index].to_dict()})

        st.session_state.results = sorted(all_results, key=lambda x: x['final_score'], reverse=True)
        st.session_state.total_matches = len(st.session_state.results)
        
        if st.session_state.total_matches > 0:
            st.success(f"Analysis complete. Found {st.session_state.total_matches} potential matches above the {score_threshold} threshold.")
        
        st.rerun()

# --- Display Logic ---
if 'results' in st.session_state:
    # If the search ran but found no results, show a message.
    if not st.session_state.results and st.session_state.get('total_matches') == 0:
        st.warning("No potential matches found for the given name and threshold.")

    if st.session_state.results:
        page = st.session_state.get('page_number', 0)
        start_idx = page * page_size
        end_idx = start_idx + page_size
        results_to_display = st.session_state.results[start_idx:end_idx]
        
        if results_to_display:
            st.subheader(f"Displaying matches {start_idx + 1} to {min(end_idx, st.session_state.total_matches)} of {st.session_state.total_matches}")
            
            with st.spinner("Getting detailed explanations from reasoning LLM..."):
                data_for_llm = [{**res['screening_data'], 'candidate_name': res['candidate_name']} for res in results_to_display]
                current_entity_type = results_to_display[0]['full_record'].get('type', 'individual')
                
                if current_entity_type == 'individual':
                    llm_assessments = get_batch_llm_assessment(data_for_llm)
                else:
                    llm_assessments = get_batch_entity_llm_assessment(data_for_llm)

            if not llm_assessments:
                st.error("Could not get explanations from the reasoning LLM.")
            else:
                for result in results_to_display:
                    assessment = llm_assessments.get(result['candidate_name'])
                    if not assessment: continue
                    
                    score = result['final_score']
                    color = "red" if score > 0.9 else "orange" if score > 0.8 else "blue"

                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1: st.markdown(f"#### Match: **{result['candidate_name']}**")
                        with col2: st.metric("Final Match Score", f"{score:.2%}", delta_color="off")

                        st.markdown("**LLM Reasoning:**")
                        st.info(assessment.get('reasoning', 'No reasoning provided.'))
                        
                        with st.expander("Show Full Analysis Dossier"):
                            screening_data = result.get('screening_data', {})
                            if screening_data.get('input_analysis'):
                                input_analysis = screening_data.get('input_analysis')
                                candidate_analysis = screening_data.get('candidate_analysis')
                                st.markdown("##### Onomastic Classification")
                                c1, c2 = st.columns(2)
                                c1.metric("Input Name Type", input_analysis.type_display_name, f"{input_analysis.engine} @ {input_analysis.probabilities.get(input_analysis.top_type_id, 0.0):.0%} conf.")
                                c2.metric("Candidate Name Type", candidate_analysis.type_display_name, f"{candidate_analysis.engine} @ {candidate_analysis.probabilities.get(candidate_analysis.top_type_id, 0.0):.0%} conf.")
                                st.markdown("##### Evidence & Scoring")
                                c1, c2, c3 = st.columns(3)
                                c1.write("**Blended Weights:**"); c1.json(screening_data.get('blended_weights_used', {}), expanded=False)
                                c2.write("**Raw Scores:**"); c2.json(screening_data.get('raw_scores', {}), expanded=False)
                                c3.metric("Data-driven Score", f"{screening_data.get('weighted_score', 0.0):.4f}")
                            else: # Entity Display Logic
                                match_details = screening_data.get('match_details', {})
                                st.markdown("##### Entity Normalization & Matching")
                                c1, c2, c3 = st.columns(3)
                                c1.text_area("Input", value=f"Original: {match_details.get('original_1', '')}\n---\nNormalized: {match_details.get('normalized_1', '')}", height=120, key=f"input_{result['candidate_name']}")
                                c2.text_area("Candidate", value=f"Original: {match_details.get('original_2', '')}\n---\nNormalized: {match_details.get('normalized_2', '')}", height=120, key=f"candidate_{result['candidate_name']}")
                                c3.metric("Normalized Match Score", f"{match_details.get('match_score', 0.0):.4f}")

                            st.markdown("##### Key Parameters from LLM")
                            for param in assessment.get('key_parameters', []): st.markdown(f"- *{param}*")
                            st.markdown("##### Full Watchlist Record"); st.json(result.get('full_record', {}), expanded=False)

        # --- Pagination logic ---
        if end_idx < st.session_state.total_matches:
            if st.button("Retrieve Additional Matches"):
                st.session_state.page_number = st.session_state.get('page_number', 0) + 1
                st.rerun()
        elif st.session_state.total_matches > 0:
            st.info("No more matches to display.")
