# app.py
# V5.9.4 (Definitive Debug Version) - Added comprehensive debug output for the 'type' column.

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
    input_name = st.text_input("Enter the full name to screen:", "DELTA PARTS SUPPLY FZC", label_visibility="collapsed")
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
            # ... (Individual logic is unchanged) ...
        else: # Logic for Entity, Vessel, Aircraft
            with st.spinner(f"Screening for non-individual entities..."):
                entity_map = {'Entity': 'entity', 'Vessel': 'vessel', 'Aircraft': 'aircraft'}
                target_type_string = entity_map.get(entity_type)
                
                # --- NEW DEBUGGING BLOCK ---
                if not st.session_state.get('debug_printed', False):
                    st.warning("One-time Debug Information:")
                    with st.expander("Click to see the unique values in the 'type' column of your data file"):
                        st.write("This will tell us if the filter string is correct. The values should be lowercase (e.g., 'entity', 'vessel').")
                        st.dataframe(sdn_df['type'].value_counts())
                    st.session_state.debug_printed = True
                
                target_df = sdn_df[sdn_df['type'] == target_type_string].copy()
                
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
# This section remains unchanged from the last correct version (V5.9.2)
if 'results' in st.session_state:
    if not st.session_state.results and st.session_state.get('total_matches') == 0:
        st.warning("No potential matches found for the given name and threshold.")
    # ... (The rest of the display logic is identical and omitted for brevity) ...
