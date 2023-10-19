import json

import pandas as pd
import requests
from tqdm import tqdm
import llm_interface


def load_prompt():
    with open("prompts/llama_2_template.txt") as f:
        prompt_template = f.read()
    with open("prompts/augmentation.txt") as f:
        prompt_augmentation_task = f.read()

    prompt = prompt_template.replace("[TASK_PROMPT]", prompt_augmentation_task)

    return prompt


def main():
    # Our goal is to augment the offers with relevant keywords using LLMs
    # Currently this code uses a Locally ran instance of LLama 2 70b q4, but the end point can be replaced with any
    # other LLM endpoint such as ChatGPT.

    # The cost of augmenting the data is very minimal and is a one time cost.

    brand_category_df = pd.read_csv("data/brand_category.csv")
    categories_df = pd.read_csv("data/categories.csv")
    offer_retailer_df = pd.read_csv("data/offer_retailer.csv")

    brand_category_df = brand_category_df[["BRAND", "BRAND_BELONGS_TO_CATEGORY"]]

    # left join on Brand
    merged_df = offer_retailer_df.merge(brand_category_df, on="BRAND", how="left")

    # Left join on 'BRAND_BELONGS_TO_CATEGORY' and "PRODUCT_CATEGORY"
    merged_df = merged_df.merge(categories_df,
                                left_on="BRAND_BELONGS_TO_CATEGORY",
                                right_on="PRODUCT_CATEGORY",
                                how="left")

    # merge PRODUCT_CATEGORY and IS_CHILD_CATEGORY_TO into a single column separated by a comma
    merged_df["CATEGORY"] = merged_df["PRODUCT_CATEGORY"] + ", " + merged_df["IS_CHILD_CATEGORY_TO"]

    merged_df = merged_df[["OFFER", "RETAILER", "BRAND", "CATEGORY"]]
    merged_df = merged_df.fillna("")

    # merge categories into a single column separated by a comma
    merged_df = merged_df.groupby(["OFFER", "RETAILER", "BRAND"])["CATEGORY"].apply(
        lambda x: ', '.join(x)).reset_index()

    prompt_template = load_prompt()

    keywords_array = []
    for i, row in tqdm(merged_df.iterrows()):

        task_str = f"Offer: {row['OFFER']}\n" \
                   f"Retailer: {row['RETAILER']}\n" \
                   f"Brand: {row['BRAND']}\n" \
                   f"Categories: {row['CATEGORY']}\n"
        current_prompt = prompt_template.replace("[TASK]", task_str)

        response, success = llm_interface.get_llm_response(current_prompt)
        if success:
            keywords_array.append(response)
        else:
            print(f"Error: {response}")

    merged_df["KEYWORDS"] = keywords_array

    merged_df.to_csv("data/offers_augmented.csv", index=False)


if __name__ == "__main__":
    main()
