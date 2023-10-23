import json
import os
import pickle

import pandas as pd
import streamlit as st


def load_prompt():
    with open("prompts/llama_2_template.txt") as f:
        prompt_template = f.read()
    with open("prompts/keyword_expansion.txt") as f:
        prompt_augmentation_task = f.read()

    prompt = prompt_template.replace("[TASK_PROMPT]", prompt_augmentation_task)

    return prompt


def load_tf_idf_model(name):
    with open(f'models/{name}_model.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'models/{name}_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    return vectorizer, tfidf_matrix


def load_llm_cache():
    model_path = "models/llm_cache.json"

    llm_cache = {}
    if os.path.exists(model_path):
        with open(model_path) as f:
            llm_cache = json.load(f)
    return llm_cache


def save_llm_cache(llm_cache):
    model_path = "models/llm_cache.json"

    with open(model_path, "w") as f:
        json.dump(llm_cache, f, indent=2)


def get_recommendation(search_query, offers_df,
                       vectorizer, tfidf_matrix,
                       vectorizer_augmented, tfidf_matrix_augmented,
                       llm_cache):
    embedded_search_query = vectorizer.transform([search_query]).toarray()
    similarity_score = tfidf_matrix.dot(embedded_search_query[0])
    top_5_indices = similarity_score.argsort()[-5:][::-1]

    if similarity_score[top_5_indices[0]] > 0.3:
        selected_indices = top_5_indices
        selected_similarity_score = similarity_score
    else:
        print("Using the alternative model")
        if search_query in llm_cache:
            search_query_augmented = llm_cache[search_query]
        else:
            search_query_augmented = search_query

        embedded_augmented_search_query = vectorizer_augmented.transform([search_query_augmented]).toarray()
        similarity_score_augmented = tfidf_matrix_augmented.dot(embedded_augmented_search_query[0])
        top_5_indices_augmented = similarity_score_augmented.argsort()[-5:][::-1]

        selected_indices = top_5_indices_augmented
        selected_similarity_score = similarity_score_augmented

    selected_offers = offers_df.iloc[selected_indices].reset_index(drop=True)
    selected_offers_score = selected_similarity_score[selected_indices]

    offer_lines = []

    for idx, row in selected_offers.iterrows():
        current_offer_text = f"{selected_offers_score[idx]:.2f} {row['OFFER']}"
        offer_lines.append(current_offer_text)

    offers_text = "\n".join(offer_lines)

    return offers_text


def main():
    vectorizer, tfidf_matrix = load_tf_idf_model("tfidf")
    vectorizer_augmented, tfidf_matrix_augmented = load_tf_idf_model("tfidf_augmented")
    llm_cache = load_llm_cache()
    offers_df = pd.read_csv("data/offers_augmented.csv")

    def on_text_change():
        # get the current text
        search_query = st.session_state.search_input
        offers_str = get_recommendation(search_query, offers_df,
                                        vectorizer, tfidf_matrix,
                                        vectorizer_augmented, tfidf_matrix_augmented,
                                        llm_cache)
        # update the offers text area
        st.session_state.offers_input = offers_str

    title = st.text_input('Search', '', on_change=on_text_change, key='search_input')
    offers_text = st.text_area('Offers', '', key='offers_input', height=500)


if __name__ == "__main__":
    main()
