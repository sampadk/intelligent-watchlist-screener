import streamlit as st
import pandas as pd
from screening_logic import load_data, calculate_all_scores, get_cosine_similarity
from llm_handler import get_llm_assessment

st.set_page_config(layout="wide")
st.title("🤖 Explainable Watchlist Screening AI")

# Load data using Streamlit's cache for efficiency
sdn_df = st.cache_data(load_data)()

# --- UI Elements ---
input_name = st.text_input("Enter the full name of the individual or entity to screen:", "Mohammad Al-Hamid")
threshold = st.slider("Set a minimum Jaro-Winkler score to trigger LLM analysis:", 0.7, 1.0, 0.85)
screen_button = st.button("Screen Name")

# --- Logic on Button Click ---
if screen_button:
    if not input_name:
        st.warning("Please enter a name to screen.")
    else:
        with st.spinner("Screening... This may take a moment."):
            # 1. First Pass: Get potential matches using fast Jaro-Winkler
            potential_matches = sdn_df[sdn_df['clean_name'].apply(
                lambda name: jellyfish.jaro_winkler_similarity(input_name.lower(), str(name)) >= threshold
            )]

            if potential_matches.empty:
                st.success(f"✅ No potential matches found above the {threshold} Jaro-Winkler threshold.")
            else:
                st.info(f"Found {len(potential_matches)} potential matches. Now calculating detailed scores and running LLM analysis...")

                results = []
                # Prepare for Cosine Similarity
                name_list = potential_matches['clean_name'].tolist()
                cosine_scores = get_cosine_similarity(input_name.lower(), name_list)

                for index, row in potential_matches.iterrows():
                    matched_name = row['name']
                    
                    # This now returns a dictionary with Jaro, Levenshtein, AND TheFuzz
                    scores = calculate_all_scores(input_name, matched_name)
                    
                    # Now we add the final, fourth score to the dictionary
                    list_index = name_list.index(row['clean_name'])
                    scores['cosine_similarity'] = round(cosine_scores[list_index], 3)
                    
                    # 3. Third Pass: Send the COMPLETE package of 4 scores to the LLM
                    llm_result = get_llm_assessment(input_name, matched_name, scores)

                    if llm_result:
                        results.append({
                            "matched_name": matched_name,
                            "scores": scores,
                            "llm_result": llm_result,
                            "full_record": row.to_dict()
                        })
                # 4. Display Results
                st.subheader("Screening Results:")
                
                # Sort results by the LLM's risk score
                sorted_results = sorted(results, key=lambda x: x['llm_result']['combined_risk_score'], reverse=True)

                for result in sorted_results:
                    score = result['llm_result']['combined_risk_score']
                    color = "red" if score > 85 else "orange" if score > 70 else "blue"
                    
                    with st.container(border=True):
                        st.markdown(f"#### Match: **{result['matched_name']}**")
                        st.markdown(f"##### <span style='color:{color};'>LLM Combined Risk Score: {score}</span>", unsafe_allow_html=True)
                        st.write("**LLM Explanation:**")
                        st.info(result['llm_result']['summary_explanation'])
                        
                        with st.expander("Show detailed scores and record"):
                            st.json(result['scores'])
                            st.write("**Full Watchlist Record:**")
                            st.json(result['full_record'])