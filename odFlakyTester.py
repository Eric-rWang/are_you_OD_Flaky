import openai
import os
import ast
import glob
import networkx as nx
import matplotlib.pyplot as plt

# Read the API key from a text file
with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip().split('=')[1].strip()

openai.api_key = api_key

def analyze_code_for_order_dependency(code_block):
    # The prompt to evaluate if the code is an order-dependent flaky test
    prompt = f"""
Here is a block of code representing unit tests:

{code_block}

Please analyze the following:
1. Are there any shared variables or state that could cause order dependency between test methods?
2. Is this unit test an order-dependent flaky test? Explain why or why not.
3. If the code is order-dependent, how can we fix it?
4. If you are unsure, ask the user to provide more information.
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

def extract_test_functions_from_file(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)

    test_functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            function_code = ast.get_source_segment(open(file_path).read(), node)
            test_functions.append((node.name, function_code))

    return test_functions

# Define the path to the testing files
testing_files_path = 'testingFiles/Breathe/*.py'

# Get all Python test files
test_files = glob.glob(testing_files_path)

# Create a directed graph to represent dependencies
dependency_graph = nx.DiGraph()

# Iterate over each test file and extract test functions
for test_file in test_files:
    print(f"Analyzing test file: {test_file}\n")
    test_functions = extract_test_functions_from_file(test_file)
    for test_name, test_function in test_functions:
        print(f"Analyzing test function: {test_name}\n{test_function}\n")
        result = analyze_code_for_order_dependency(test_function)
        print(result)
        print("\n" + "-" * 50 + "\n")

        # Add nodes and dependencies to the graph based on the analysis
        if "order-dependent" in result.lower():
            dependency_graph.add_node(test_name)
            # Placeholder: For demonstration purposes, assume each flaky test depends on the previous one
            # You can modify this logic to reflect actual dependencies based on the analysis
            if len(dependency_graph.nodes) > 1:
                previous_test = list(dependency_graph.nodes)[-2]
                dependency_graph.add_edge(previous_test, test_name)

# Visualize the dependency graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(dependency_graph)
nx.draw(dependency_graph, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=6, font_weight='bold')
plt.title('Order-Dependency Relationships Between Test Functions')
plt.show()
