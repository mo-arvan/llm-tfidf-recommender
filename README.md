# Offer Recommendation System

## Introduction

Given a list of offers and some associated metadata around the retailers and brands that are sponsoring the offer, we
must provide a tool that allows users to search for offers via text input from the user.

## Approach

Skip to the Offer Recommendation to see the final result if you are not interested in the details of the approach.

### 0. Environment Setup

Build the docker image and run the container. The container will be used to run the code. The code is written in Python

```bash
docker build -t offer-recommender -f docker/slim.dockerfile .
```

Run it with:

```bash 
docker run -it --rm -v $(pwd):/app offer-recommender bash
```

### 1. Offer Augmentation

We will merge the offers and the metadata into a single dataframe. The metadata does not seem to contain much
information, therefore, we will use all the available information for each offer and pass it to an LLM to augment our
input with additional keywords.

Note that this step cannot be performed without direct access to an LLM. Currently, the code users a privately ran
instance of Llama 2 70b q4 using text generation inference from Huggingface. The endpoint APIs are similar to ChatGPT,
therefore, the code can be easily modified to use ChatGPT instead. Using Llama 2 with this level of quantization is not
ideal and will result in suboptimal results, nonetheless, it is sufficient for the purpose of building a proof of
concept.

Run the below command to generate the augmented input. The augmented input will be saved in `data/offers_augmented.csv`.

```bash
python src/llm_augmentation.py
```

### 2. Recommendation Model

We will use a simple tf-idf model to recommend offers to the user. The tf-idf model will be trained on the augmented
input. The simplicity and low computational cost of the tf-idf model makes it a good candidate for this task. In the
later stages, we can utilize cosine similarity to recommend offers to the user.

```bash
python src/recommender_model.py
```

### 3. Offer Recommendation

We will use the trained tf-idf model to recommend offers to the user. The values are fixed at the moment, but again,
this is a proof of concept. The code uses tfidf model trained on offers and the metadata first. If the similarity score
is below a certain threshold, it will use the tfidf model trained on the augmented input.

```bash
python src/offer_recommender.py
```

### Evaluation

The file [output.txt](results%2Foutput.txt) contains the outputs of the model. We will explore the results to ensure
that the model is working as expected. Ultimately, we need a way to evaluate this work properly to determine the best
approach. The recommendations are also provided in [results.csv](results%2Fresults.csv).

### Top 1 for Target

The results look good, expanded keywords looks useful, but the expanded model has not been used due to high score for
the original model.

```text
Top 5 offers for Target
Expanded keywords: Affordable fashion Home goods Electronics Toys Baby products Clothing Shoes Accessories Home decor Furniture Outdoor gear Sports equipment Pet supplies Health and beauty Personal care Household essentials
Arber, at Target
Score: 0.5517991524338901
Retailer: TARGET
Brand: ARBER
Categories: nan
Keywords:   "Arber", "Target", "outdoor furniture", "patio sets", "garden decor", "home decor", "furniture"

```

### Top 1 for Huggies

This time, we are using the augmented keywords. The results are average, I looked for potential matches in the offers
and I couldn't find any relevant offers at first glance. I suppose relevant offers targeting baby product are relevant,
this needs to be evaluated further.

```text
Using augmented keywords
Top 5 offers for Huggies
Expanded keywords: Diapers Baby wipes Nappies Disposable diapers Baby care Parenting Infant products Toddler products Baby hygiene Diapering essentials Changing supplies Nursery products
Spend $220 at Tom Thumb
Score: 0.2916540836341517
Retailer: TOM THUMB
Brand: TOM THUMB
Categories: nan
Keywords:   "Grocery, Supermarket, Food, Beverages, Household Essentials, Personal Care, Pet Supplies, Baby Products"

```




