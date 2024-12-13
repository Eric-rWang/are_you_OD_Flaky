import pandas as pd
import os
import re
import ast
import json
from openai import OpenAI

with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip().split('=')[1].strip()

client = OpenAI(api_key=api_key)

def get_diff_code(file_path, patch, function_name):

    # Read the file content
    with open(file_path, "r") as f:
        file_content = f.read()

    # Convert the file content to a list of lines
    lines = file_content.splitlines()

    # Parse the patch
    patch_lines = patch.strip().split("\n")
    match = re.match(r"(\d+)([acd])(\d*,\d*)?", patch_lines[0])
    if not match:
        raise ValueError(f"Invalid patch format: {patch_lines[0]}")

    line_num, command, _ = match.groups()
    line_num = int(line_num) - 1  # Convert to 0-based index

    # Extract the content to add from the patch
    content_lines = [line.lstrip("> ") for line in patch_lines[1:]]

    if command == "a":  # Add content after the line
        for i, content_line in enumerate(content_lines):
            lines.insert(line_num + 1 + i, content_line)
    elif command == "c":  # Change content at the line (replace)
        # Remove affected lines and replace with patch content
        range_match = re.search(r"(\d+),(\d+)", patch_lines[0])
        if range_match:
            start, end = map(int, range_match.groups())
            del lines[start - 1:end]
        lines.insert(line_num, "\n".join(content_lines))
    elif command == "d":  # Delete lines
        range_match = re.search(r"(\d+),(\d+)", patch_lines[0])
        if range_match:
            start, end = map(int, range_match.groups())
            del lines[start - 1:end]
        else:
            lines.pop(line_num)
    else:
        raise ValueError(f"Unknown patch command: {command}")

    # Join the updated lines to form the final content
    diff_applied = "\n".join(lines)

    messages = [ {"role": "system", "content": "You are an expert in computer science."} ]
    prompt = f"""
        Can you please fix the indentations for the following code and format it? With the formated code, 
        can you return the function {function_name}. Please do not provide additional text/explaination, 
        respond in plain text and do not add '''python ''' to the code. Please remove the "```python" markdown 
        snippet from your response, again plain text only.
        {diff_applied}
    """
    messages.append(
        {"role": "user", "content": prompt},
    )
    chat = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=messages
    )

    reply = chat.choices[0].message.content
    reply = re.sub(r'^```.*\n|\n```$', '', reply)

    print(reply)

    tree = ast.parse(reply)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            function_code = ast.get_source_segment(reply, node)
            test_name = node.name
            if test_name == function_name: # function we are looking for
                # print(f"Analyzing test function: {test_name}\n{function_code}\n")
                # need to apply diff...
                print('Diff code:\n', function_code)
                return function_code
            else:
                continue

    return 0 


def get_file_content(file_path, function_name):
        
    # Read the file containing the function
    with open(file_path, "r") as f:
        # file_content = f.read()
        tree = ast.parse(f.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
            function_code = ast.get_source_segment(open(file_path).read(), node)
            test_name = node.name
            if test_name == function_name: # function we are looking for
                # print(f"Analyzing test function: {test_name}\n{function_code}\n")
                # need to apply diff...
                print('Original code:\n', function_code)
                return function_code
            else:
                continue

def process_excel_file(file_path, base_folder="flakyProjects"):
    """
    Process an Excel file containing patch data to output original and patched functions.

    Parameters:
        file_path (str): Path to the Excel file.
        base_folder (str): Path to the folder containing cloned projects.

    Outputs:
        Prints the original and patched functions for each entry in the Excel file.
    """
    df = pd.read_excel(file_path)[::-1]
    output_data = []
    
    for index, row in df.iterrows():
        project_name = row["Project_Name"]
        test_id = row["Test_id"]
        diff = row["Diff"]
        diff_path = row["Path"]

        # Parse the test_id
        test_id_parts = test_id.split("::")
        file_path = os.path.join(base_folder, project_name, test_id_parts[0])
        
        # Extract class and function names
        if len(test_id_parts) == 2:
            class_name = None
            function_name = test_id_parts[1]
        elif len(test_id_parts) == 3:
            class_name = test_id_parts[1]
            function_name = test_id_parts[2]
        else:
            print(f"Unexpected test_id format: {test_id}")
            continue

        # Remove content in square brackets from function name
        function_name = re.sub(r"\[.*\]", "", function_name)

        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Skipping test {test_id}.")
            continue
        
        try:
            function_code = get_file_content(file_path, function_name)
            diff_code = get_diff_code(file_path, diff, function_name)

            output_data.append({
                "project_name": project_name,
                "flaky_code": function_code,
                "fixed_code": diff_code
            })

            print(f'Finished {project_name}, {function_name}')

        except Exception as e:
            print(f"Error while processing {project_name}, {function_name}: {e}")
            continue
        
    output_file = "flakyTestBank.json"
    with open(output_file, "w") as file:
        json.dump(output_data, file, indent=4)

    print(f"Output written to {output_file}")


# Example usage
process_excel_file("./clear/Patches.xlsx", base_folder="./flakyProjects")
