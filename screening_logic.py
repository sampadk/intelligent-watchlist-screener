import pandas as pd
import jellyfish
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import fuzz # <-- IMPORT THEFUZZ

# Load the sanctions list ONCE when the app starts
# The @st.cache_data in Streamlit will make this efficient
def load_data():
    df = pd.read_csv("sdn.csv", header=None)
    # Give columns meaningful names - check the OFAC data dictionary for exact columns
    # For this example, let's assume the name is in the second column (index 1)
    df.columns = ['ent_num', 'name', 'type', 'program', 'title', 'call_sign', 'vess_type', 'tonnage', 'grt', 'vess_flag', 'vess_owner', 'remarks']
    df['clean_name'] = df['name'].str.lower().str.strip()
    return df

# Function to calculate all scores between two names
def calculate_all_scores(name1, name2):
    name1 = str(name1).lower()
    name2 = str(name2).lower()

    # 1. Jaro-Winkler from jellyfish
    jaro_score = jellyfish.jaro_winkler_similarity(name1, name2)

    # 2. Levenshtein (normalized) from jellyfish
    lev_dist = jellyfish.levenshtein_distance(name1, name2)
    max_len = max(len(name1), len(name2))
    lev_score = (1 - lev_dist / max_len) if max_len > 0 else 0
    
    # 3. TheFuzz Token Set Ratio <-- ADDED THIS SCORE
    # This score is powerful for names. We divide by 100 to normalize it (0 to 1).
    fuzz_score = fuzz.token_set_ratio(name1, name2) / 100

    return {
        "jaro_winkler": round(jaro_score, 3),
        "levenshtein_normalized": round(lev_score, 3),
        "thefuzz_token_set_ratio": round(fuzz_score, 3) # <-- ADDED TO OUTPUT
    }

# Special function for Cosine Similarity as it works on a list of names
def get_cosine_similarity(input_name, name_list):
    documents = [input_name] + name_list
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    
    # The first vector is our input_name, compare it to all others
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return cosine_sim[0] # Returns an array of scores