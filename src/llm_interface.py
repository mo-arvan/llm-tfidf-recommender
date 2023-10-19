import json
import re

import requests


def get_llm_response(prompt_text):
    server_url_dict = {
        "local": "http://192.168.100:3000/generate",
        "remote": "http://10.8.48.25:3000/generate",

    }
    url = server_url_dict["remote"]
    data = {
        'inputs': prompt_text,
        'parameters': {'max_new_tokens': 256,  # The maximum numbers of new tokens
                       'temperature': 0.2,  # Higher values produce more diverse outputs, Range: [0, 1]
                       'top_p': 0.95,  # Higher values sample more low-probability tokens, Range: [0, 1]
                       'repetition_penalty': 1.1,  # Penalize repeated tokens, 1.0 means no penalty, Range: [1, 2]
                       # 'repetition_penalty_range': 512,
                       }
    }
    headers = {'Content-Type': 'application/json'}

    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        result = response.json()
        generated_text = result['generated_text']
    else:
        generated_text = f"Error: {response.status_code} - {response.text}"
        print(generated_text)

    return generated_text, response.status_code == 200


def clean_keyword(keyword_str):
    clean_pattern = r'\*\s*([\s\S]*?)$'

    # input_str = keyword_str.split("\n")[-1]

    all_matches = re.findall(clean_pattern, keyword_str, re.MULTILINE)

    all_matches_joined = " ".join(all_matches)

    return all_matches_joined.strip()

