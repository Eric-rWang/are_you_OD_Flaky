import os
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import numpy as np

# Set up OpenAI API key
with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip().split('=')[1].strip()

client = OpenAI(api_key=api_key)

# Pretrained model for embeddings
tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
model = AutoModel.from_pretrained("microsoft/unixcoder-base")

test_bank = [
    # General Example 1: Resetting shared state
    {
        "flaky_test": """
        def test_shared_counter():
            global_counter = 0
            global_counter += 1
            assert global_counter == 1
        """,
        "fixed_test": """
        def test_shared_counter():
            global_counter = 0
            global_counter += 1
            assert global_counter == 1
        """,
        "category": "Order Dependency",
        "keywords": ["shared state", "global"]
    },
    # General Example 2: Sorting data before processing
    {
        "flaky_test": """
        def test_unsorted_list():
            data = [3, 1, 2]
            process(data)
            assert data == [1, 2, 3]
        """,
        "fixed_test": """
        def test_unsorted_list():
            data = [3, 1, 2]
            data.sort()
            process(data)
            assert data == [1, 2, 3]
        """,
        "category": "Order Dependency",
        "keywords": ["sorting", "unordered"]
    },
    # General Example 3: Timing issues and non-deterministic behavior
    {
        "flaky_test": """
        def test_random_sleep():
            start = time.time()
            time.sleep(random.uniform(0.1, 0.5))
            elapsed = time.time() - start
            assert elapsed > 0.1
        """,
        "fixed_test": """
        def test_random_sleep():
            random.seed(42)
            start = time.time()
            time.sleep(0.3)
            elapsed = time.time() - start
            assert elapsed > 0.1
        """,
        "category": "Order Dependency",
        "keywords": ["timing", "random"]
    },
    # General Example 4: Clearing state between tests
    {
        "flaky_test": """
        def test_accumulator():
            accumulator.append(1)
            assert sum(accumulator) == 1
        """,
        "fixed_test": """
        def test_accumulator():
            accumulator = []
            accumulator.append(1)
            assert sum(accumulator) == 1
        """,
        "category": "Order Dependency",
        "keywords": ["state management", "cleanup"]
    },
    # General Example 5: Handling external resources
    {
        "flaky_test": """
        def test_temp_file_creation():
            with open("temp.txt", "w") as f:
                f.write("test")
            assert os.path.exists("temp.txt")
        """,
        "fixed_test": """
        def test_temp_file_creation():
            with open("temp.txt", "w") as f:
                f.write("test")
            assert os.path.exists("temp.txt")
            os.remove("temp.txt")
        """,
        "category": "Order Dependency",
        "keywords": ["external resources", "file handling", "cleanup"]
    },
    # General Example 6: Mocking external state
    {
        "flaky_test": """
        def test_environment_variable():
            os.environ["MY_VAR"] = "test"
            assert os.getenv("MY_VAR") == "test"
        """,
        "fixed_test": """
        def test_environment_variable():
            os.environ["MY_VAR"] = "test"
            try:
                assert os.getenv("MY_VAR") == "test"
            finally:
                del os.environ["MY_VAR"]
        """,
        "category": "Order Dependency",
        "keywords": ["mocking", "environment variable"]
    }
]



# Helper: Get Code Embedding
def get_code_embedding(code):
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    # Reduce to 2D by averaging across sequence length (axis 1)
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

# Helper: Find Similar Tests
def find_similar_tests(test_code, test_bank, top_k=3):
    test_embedding = get_code_embedding(test_code).reshape(1, -1)  # Ensure 2D for cosine_similarity
    bank_embeddings = [get_code_embedding(entry["flaky_test"]).reshape(1, -1) for entry in test_bank]
    bank_embeddings = np.vstack(bank_embeddings)  # Stack all embeddings into a single 2D array

    similarities = cosine_similarity(test_embedding, bank_embeddings)[0]
    ranked_tests = sorted(zip(test_bank, similarities), key=lambda x: x[1], reverse=True)
    
    # Return both test bank entries and their similarity scores
    return [(entry[0], entry[1]) for entry in ranked_tests[:top_k]]


# Helper: Construct Prompt
def construct_order_dependent_prompt(test_code, similar_tests):
    prompt = f"""
        You are a Python test expert specializing in fixing flaky tests. 
        The test below is flaky due to order dependency. Please comment 
        the code with the fixes made and list out the reasons for the 
        change. Be brief and concise with your explanations. Thank you!

        Here is the flaky test code:
        {test_code}

        Here are some examples of flaky tests and their fixes:
    """

    for test in similar_tests:
        prompt += f"""
            Example Flaky Test:
            {test['flaky_test']}

            Fixed Test:
            {test['fixed_test']}
        """

    prompt += "\nNow provide the fixed test code for the given flaky test."

    return prompt

# Repair Flaky Test Using GPT
def repair_flaky_test(flaky_test_code, fix_category, test_bank):
    """
    Prompts GPT-3.5 to repair a flaky test.

    Args:
        flaky_test_code (str): The flaky test code to be fixed.
        fix_category (str): The predicted category of the fix (e.g., "Order Dependency").
        test_bank (list): List of examples with flaky tests and fixes.

    Returns:
        str: The repaired test code suggested by GPT.
    """
    # Find similar tests
    similar_tests_with_scores = find_similar_tests(flaky_test_code, test_bank)
    
    print("Similar tests picked with scores:")
    for test, score in similar_tests_with_scores:
        print(f"Similarity Score: {score:.4f}")
        print(f"Flaky Test:\n{test['flaky_test']}")
        print(f"Fixed Test:\n{test['fixed_test']}\n")

    # Extract only the tests for constructing the prompt
    similar_tests = [test for test, score in similar_tests_with_scores]

    # Construct the prompt
    prompt = construct_order_dependent_prompt(flaky_test_code, similar_tests)

    # Call the OpenAI GPT API
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Python expert in order-dependent flaky test detection."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        repaired_code = response.choices[0].message.content
        return repaired_code
    except Exception as e:
        print(f"Error while generating repair: {e}")
        return None


# Main Repair Pipeline
if __name__ == "__main__":

    # Flaky test
    flaky_test_code = """
        def test_execution_order():
            results = []
            results.append(part1())
            results.append(part2())
            assert results == [1, 2]
    """

    # Fix category
    fix_category = "Order Dependency"

    # Repair the test
    print("Repairing Flaky Test...")
    repaired_code = repair_flaky_test(flaky_test_code, fix_category, test_bank)
    print("Repaired Test Code:\n", repaired_code)
