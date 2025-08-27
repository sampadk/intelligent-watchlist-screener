# app.py
# V5.8 - Added Reset button, decoupled controls, simpler button name, and list selector.

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
    st.session_state.clear()

# --- UI Controls in Sidebar ---
with st.sidebar:
    st.header("⚙️ Screening Controls")
    
    # d) List selector (pre-selected and disabled for now)
    available_lists = ['OFAC Consolidated']
    selected_lists = st.multiselect(
        "Select Sanctions Lists",
        options=available_lists,
        default=available_lists,
        disabled=True, # Will be enabled when more lists are added
        help="Currently locked to the OFAC list. More lists will be added in future iterations."
    )
    
    entity_type = st.selectbox("Select Entity Type", ('Individual', 'Entity', 'Vessel', 'Aircraft'))
    score_threshold = st.slider("Match Score Threshold", 0.5, 1.0, 0.7, 0.01, help="Only show matches with a final data-driven score above this value.")
    page_size = st.slider("Matches per batch", 1, 10, 5)

# --- Main Search Area ---
col1, col2, col3 = st.columns([4, 1, 1])
with col1:
    input_name = st.text_input("Enter the full name to screen:", "KIM Jong Un", label_visibility="collapsed")
with col2:
    # c) Simpler button name
    if st.button("Screen Name", type="primary", use_container_width=True):
        st.session_state.run_search = True # Set a flag to run the search
with col3:
    # a) Reset button
    st.button("Reset", on_click=reset_search, use_container_width=True)

# b) Decouple search logic to only run when the button is clicked
if st.session_state.get('run_search', False):
    st.session_state.run_search = False # Reset the flag
    st.session_state.results = []
    st.session_state.page_number = 0
    
    if not input_name:
        st.warning("Please enter a name.")
    else:
        # (The rest of the search logic remains the same as V5.7)
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
                
                for row in target_df.itertuples():
                    match_details = normalize_and_match_entity(input_name, row.name)
                    if match_details['match_score'] >= score_threshold:
                        all_results.append({"candidate_name": row.name, "final_score": match_details['match_score'], "screening_data": {"input_name": input_name, "candidate_name": row.name, "match_details": match_details}, "full_record": sdn_df.loc[row.Index].to_dict()})

        st.session_state.results = sorted(all_results, key=lambda x: x['final_score'], reverse=True)
        st.session_state.total_matches = len(st.session_state.results)
        st.success(f"Analysis complete. Found {st.session_state.total_matches} potential matches above the {score_threshold} threshold.")
        st.rerun() # Rerun to ensure display logic uses the new state

# --- Display Logic (now independent of the search button press) ---
if 'results' in st.session_state and st.session_state.results:
    # (The entire display logic from V5.7 goes here and is unchanged)
    # ... (omitted for brevity, it's identical to the previous version's display loop)

# (The pagination logic is also unchanged)
