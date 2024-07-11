"""
This module generates synthetic data examples and appends them to an existing dataset.
"""

import os
import json
import pandas as pd
import openai

with open("prompt.txt", encoding="utf-8") as f:
    prompt = f.read()

# Initialize OpenAI client
client = openai.OpenAI(
    api_key=os.getenv('OPEN_AI_KEY')
)

def generate_synthetic_example(prompt_text):
    """
    Generates a synthetic example based on the given prompt using the OpenAI API.

    Args:
        prompt_text (str): The prompt text to generate the synthetic example.

    Returns:
        dict or None: The generated synthetic example if successful, otherwise None.
    """
    required_keys = ["email", "response"]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at generating synthetic data.",
                },
                {"role": "user", "content": prompt_text},
            ],
        )
        # Correctly accessing the message content from the API response
        response_json = json.loads(response.choices[0].message.content)

        if set(response_json.keys()) == set(required_keys):
            return response_json
        return None
    except Exception as error:
        print(f"Error generating synthetic example: {error}")
        return None


NUMBER_OF_SAMPLES = 500
synthetic_examples = []
for i in range(NUMBER_OF_SAMPLES):
    example = generate_synthetic_example(prompt)
    if example:
        synthetic_examples.append(example)

dataframe = pd.DataFrame(synthetic_examples)
dataframe.to_csv("synthetic_examples.csv")
