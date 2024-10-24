import openai
import os

# Read the API key from a text file
with open('local_env.txt', 'r') as file:
    api_key = file.readline().strip().split('=')[1].strip()

openai.api_key = api_key

def analyze_code_for_flakiness(code_block):
    # The prompt to evaluate if the code is a flaky test
    prompt = f"""
        Here is a block of code representing a unit test:

        {code_block}

        Please answer the following questions:
        1. Is this unit test an order-dependent flaky test? Explain why or why not.
        2. If the code is order-dependent, how can we fix it?
        3. If the code is not a flaky test or you are unsure, ask the user to provide more information.
    """

    # Call the OpenAI API to get the response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert software engineer specializing in unit tests."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    # Extract the response content
    answer = response['choices'][0]['message']['content']
    return answer

# Example code block for analysis
test_code = """
def test_addition():
    global counter
    counter += 1
    assert add(2, 3) == 5
"""

# Run the analysis
result = analyze_code_for_flakiness(test_code)
print(result)
