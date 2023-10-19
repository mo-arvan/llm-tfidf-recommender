import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_keyword(keyword_str):
    clean_pattern = r'[\s\S]*?\"([\s\S]*?)\"'

    match = re.match(clean_pattern, keyword_str)
    if match:
        return match.group(1)
    else:
        # print(f"Warning: {keyword_str}")
        return keyword_str.strip()


def build_tfidf_model(offer_augmented_text_list, name):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(offer_augmented_text_list).toarray()

    with open(f'models/{name}_model.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'models/{name}_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)


def main():
    data_augmented_df = pd.read_csv("data/offers_augmented.csv")
    data_augmented_df["KEYWORDS"] = data_augmented_df["KEYWORDS"].apply(clean_keyword)
    data_augmented_df = data_augmented_df.fillna("")


    brand_retailer_category_list = data_augmented_df[["RETAILER", "BRAND", "CATEGORY"]].apply(
        lambda x: " ".join(x), axis=1).tolist()
    offer_text_list = data_augmented_df[["OFFER", "RETAILER", "BRAND", "CATEGORY"]].apply(
        lambda x: " ".join(x), axis=1).tolist()

    offer_augmented_text_list = data_augmented_df.apply(lambda x: " ".join(x), axis=1).tolist()

    build_tfidf_model(brand_retailer_category_list, "tfidf_brc")
    build_tfidf_model(offer_text_list, "tfidf")
    build_tfidf_model(offer_augmented_text_list, "tfidf_augmented")


if __name__ == "__main__":
    main()
