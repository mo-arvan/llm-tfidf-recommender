# Requirements and Initial Plan

## Stories

- Users mush be able to seek offers, examples:
	- If a user searches for a category (ex. diapers) the tool should return a list of offers that are relevant to that category
	- If a user searches for a brand (ex. Huggies) the tool should return a list of offers that are relevant to that brand.
	- If a user searches for a retailer (ex. Target) the tool should return a list of offers that are relevant to that retailer.
	- The tool should also return the score that was used to measure the similarity of the text input with each offer


## Non-functional requirements

- easily seek out offers
- Quality of recommendations is important, since we like to promote our partners
- "intelligent search" for offers via text input from the user

## Model

- Create a list of keywords for all categories, brands, retailers.
- Find named entities within offers, generate list of keywords assosicated with them. 


## Pipeline

1. user inputs text
2. program checks whether there is an exact match of the given keyword. 
	If not, find several alternatives to suggest to the user. 
3. measure similarity input and offers
