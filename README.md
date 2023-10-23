# Offer Recommendation System

## Introduction

Given a list of offers and some associated metadata around the retailers and brands that are sponsoring the offer, we
must provide a tool that allows users to search for offers via text input from the user.

## Approach

Skip to the Offer Recommendation to see the final result if you are not interested in the details of the approach.

My approach is to use a simple tf-idf model to recommend offers to the users. I augment the offers with additional data
using an LLM, Llama 2 70b q4 in particular. The LLM is used to generate additional keywords for each offer. This is
because the offers are very short and do not contain much information. Then, I train a tf-idf model on the original data
and the augmented data. There is a single function that handles giving a top-5
recommendation to the user. Note that the search query itself can be augmented as well to include additional 
information. The usage of LLMs could be replaced by pre-defined rules or knowledge graphs. 
This function can become and API endpoint that can be used by the front-end to recommend
offers to the user. The performance and the results are not quantified in this work, but it can be easily done by
having access to labeled data. I prefer to finalize the full pipeline including the evaluation prior to moving forward
with a more complex model. Furthermore, the expectation of the model also must be defined, and we cannot use "good" as a
descriptor for the model. Future development can utilize this code as baseline and build on top of it. The visualization
is handled by streamlit, which is a simple and easy to use tool for building quick prototypes. The UI is not the focus
of this work, but it can be easily improved by using a more sophisticated UI framework, but that is outside my
expertise.

### 0. Environment Setup

Build the docker image and run the container. The container will be used to run the code. The code is written in Python

```bash
docker build -t offer-recommender -f docker/slim.dockerfile .
```

Run it inside docker container, map ports for streamlit.

```bash
docker run -it -p 8501:8501 --rm -v $(pwd):/app offer-recommender streamlit run src/offer_recommender.py
```

All the scripts assume the environment is properly setup.

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

You must be inside the docker container to run the code.
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
python streamlit run src/offer_recommender.py
```

### Evaluation

The file [output.txt](results%2Foutput.txt) contains the outputs of the model. We will explore the results to ensure
that the model is working as expected. Ultimately, we need a way to evaluate this work properly to determine the best
approach. The recommendations are also provided in [results.csv](results%2Fresults.csv).

### Top 5 for Target

The results look good, expanded keywords looks useful, but the expanded model has not been used due to high score for
the original model.

```text
0.55 Arber, at Target
0.42 L'Oréal Paris Makeup, spend $30 at Target
0.42 L'Oréal Paris Makeup, spend $35 at Target
0.41 L'Oréal Paris True Match Foundation at Target
0.40 L'Oreal Paris True Match Foundation at Target
```

### Top 1 for Pizza

This time, we are using the augmented keywords. The results are average, I looked for potential matches in the offers
and I couldn't find any relevant offers at first glance. I suppose relevant offers targeting baby product are relevant,
this needs to be evaluated further.

```text
0.56 Whole pizza at Casey's
0.56 Whole Pizza at Casey's 
0.56 Whole Pizza at Casey's
0.54 Whole Pizza Pie at Casey's
0.51 2 Pack OR 2 Liter AND Whole Pizza at Casey's
```




