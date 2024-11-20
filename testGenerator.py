import openai
import os
import ast
import glob
import json
import pandas as pd

# Load the Patches Excel file
patches_file = 'testingFiles/Patches.xlsx'
df_patches = pd.read_excel(patches_file)

def genTest(project_name):
    # Define the path to the testing files
    testing_files_path = 'testingFiles/%s/*.py' % project_name

    # Get all Python test files
    test_files = glob.glob(testing_files_path)

    # File to save the training data
    training_data_file = 'codeBertTraining_v2.jsonl'

    # Iterate over each test file and extract test functions
    with open(training_data_file, 'a') as train_file:
        for test_file in test_files:
            print(f"Analyzing test file: {test_file}\n")
            
            # Extract test functions from the file
            with open(test_file, 'r') as file:
                tree = ast.parse(file.read(), filename=test_file)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    function_code = ast.get_source_segment(open(test_file).read(), node)
                    test_name = node.name
                    print(f"Analyzing test function: {test_name}\n{function_code}\n")

                    # Match the test function to the Patches file
                    project_tests = df_patches[df_patches['Project_Name'] == project_name]
                    test_status_row = project_tests[project_tests['Test_id'].str.contains(test_name, regex=False)]

                    if not test_status_row.empty:
                        # Extract relevant information from the Patches file
                        od_type = test_status_row.iloc[0]['OD_Type']
                        polluter_or_setter = test_status_row.iloc[0]['Polluter_or_Setter']
                        cleaner_exists = 'Cleaner' if 'Cleaner' in test_status_row.columns and test_status_row.iloc[0]['Cleaner'] else 'No Cleaner'
                        patch_diff = test_status_row.iloc[0]['Diff']

                        # Prepare the training data prompt
                        prompt = f"""Here is a unit test:

                            {function_code}

                            The test belongs to project '{project_name}'. It is identified as an order-dependent flaky test. Details are provided:
                            - Order-Dependent Type: {od_type}
                            - Polluter or Setter: {polluter_or_setter}
                            - Cleaner: {cleaner_exists}

                            The provided solution to resolve the flaky behavior is as follows:
                            {patch_diff}
                        """
                        
                        # Save the prompt and corrected response in JSONL format for fine-tuning
                        train_data = {
                            "code": function_code,
                            "od": 1, # true for OD flaky
                            # "fix": patch_diff
                        }

                        train_file.write(json.dumps(train_data) + "\n")

                    else:
                        # Skip if the test function is not found in the Patches file
                        print(f"Test '{test_name}' not found in Patches file, not flaky.\n")

                        # Save the prompt and corrected response in JSONL format for fine-tuning
                        train_data = {
                            "code": function_code,
                            "od": 0, # false for OD flaky
                            # "fix": patch_diff
                        }

                        train_file.write(json.dumps(train_data) + "\n")


genTest('py-autodoc')