from openai import OpenAI

import os

# Read the API key from a text file
with open('api_key.txt', 'r') as file:
    api_key = file.readline().strip().split('=')[1].strip()

client = OpenAI(api_key=api_key)

def fine_tune_model(training_data_path, base_model="gpt-3.5-turbo"):
    """
    Function to automate the fine-tuning process using the OpenAI Python API.

    Parameters:
    - training_data_path: str, path to the training data file (.jsonl).
    - base_model: str, the name of the base model to be fine-tuned.
    """
    try:
        # Step 1: Upload the training file
        print("Uploading training data...")
        file_response = client.files.create(file=open(training_data_path, "rb"),
        purpose='fine-tune')
        training_file_id = file_response.id

        # Step 2: Create a fine-tune job
        print("Creating fine-tune job...")
        fine_tune_response = client.fine_tuning.jobs.create(training_file=training_file_id,
        model=base_model)

        print("Fine-tuning job successfully started.")
        print(f"Fine-tune job ID: {fine_tune_response.id}")

        return fine_tune_response.id
    
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Define the path to the training data file
    training_data_file = 'flaky_test_training_data.jsonl'

    # Check if the training data file exists
    if os.path.exists(training_data_file):
        # Start the fine-tuning process
        model_id = fine_tune_model(training_data_file, base_model="gpt-3.5-turbo")
    else:
        print(f"Training data file '{training_data_file}' not found.")

    # List all fine-tune jobs
    response = client.fine_tuning.jobs.list()

    # Print out information on fine-tuned models
    print(response)

    # Use the fine-tuned model
    response = client.chat.completions.create(
        model="ftjob-lgRplMBBqKNdJIeOU5saXGGu",
        messages="Here is a unit test:\n\n<code>\n\nIs this an order-dependent flaky test?",
        temperature=0.7,
        max_tokens=150
    )

    # Print the response
    print(response['choices'][0]['text'])

