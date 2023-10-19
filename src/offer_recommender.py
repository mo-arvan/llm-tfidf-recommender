import json
import os
import pickle

import pandas as pd

import llm_interface


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


def main():
    vectorizer, tfidf_matrix = load_tf_idf_model("tfidf")
    vectorizer_augmented, tfidf_matrix_augmented = load_tf_idf_model("tfidf_augmented")
    vectorizer_brc, tfidf_matrix_brc = load_tf_idf_model("tfidf_brc")
    llm_cache = load_llm_cache()
    offers_df = pd.read_csv("data/offers_augmented.csv")

    desired_keywords = ["Target", "Huggies", "Makeup", "Pizza"]
    desired_keywords_expanded = []
    keyword_expansion_prompt = load_prompt()
    for keyword in desired_keywords:
        if keyword in llm_cache:
            desired_keywords_expanded.append(llm_cache[keyword])
            continue

        current_prompt = keyword_expansion_prompt.replace("[TASK]", keyword)
        generated_text, success = llm_interface.get_llm_response(current_prompt)
        keywords = llm_interface.clean_keyword(generated_text)
        llm_cache[keyword] = keywords
        desired_keywords_expanded.append(keywords)

    save_llm_cache(llm_cache)
    desired_keywords_array = vectorizer.transform(desired_keywords).toarray().tolist()
    desired_keywords_array_augmented = vectorizer_augmented.transform(desired_keywords_expanded).toarray().tolist()
    # we will calculate the cosine similarity between the desired keywords and the tfidf matrix
    # we will report the similarity score for each offer
    # we will print top 5 offers with the highest similarity score

    results_list = []
    for idx, keyword_tfidf in enumerate(desired_keywords_array):
        similarity_score = tfidf_matrix.dot(keyword_tfidf)
        similarity_score_augmented = tfidf_matrix_augmented.dot(desired_keywords_array_augmented[idx])
        top_5_indices = similarity_score.argsort()[-5:][::-1]
        top_5_indices_augmented = similarity_score_augmented.argsort()[-5:][::-1]

        if similarity_score[top_5_indices[0]] < 0.1:
            print("Using augmented keywords")
            selected_indices = top_5_indices_augmented
            selected_similarity_score = similarity_score_augmented
        else:
            selected_indices = top_5_indices
            selected_similarity_score = similarity_score

        print(f"Top 5 offers for {desired_keywords[idx]}")
        print(f"Expanded keywords: {desired_keywords_expanded[idx]}")
        for i in selected_indices:
            print(f"\t{offers_df.iloc[i]['OFFER']}")
            print(f"\tScore: {selected_similarity_score[i]}")
            print(f"\tRetailer: {offers_df.iloc[i]['RETAILER']}")
            print(f"\tBrand: {offers_df.iloc[i]['BRAND']}")
            print(f"\tCategories: {offers_df.iloc[i]['CATEGORY']}")
            print(f"\tKeywords: {offers_df.iloc[i]['KEYWORDS']}")
            print("\n")
            results_list.append((desired_keywords[idx], desired_keywords_expanded[idx], offers_df.iloc[i]['OFFER'],
                                 selected_similarity_score[i], offers_df.iloc[i]['RETAILER'], offers_df.iloc[i]['BRAND'],
                                 offers_df.iloc[i]['CATEGORY'], offers_df.iloc[i]['KEYWORDS']))

    results_df = pd.DataFrame(results_list,
                              columns=['input', 'input_expanded', 'offer', 'similarity_score', 'retailer', 'brand',
                                       'category', 'keywords'])
    results_df.to_csv("results/results.csv", index=False)


if __name__ == "__main__":
    main()
