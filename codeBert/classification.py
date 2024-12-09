import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from transformers import RobertaForSequenceClassification, RobertaTokenizer

def load_model(model_dir, device):
    """
    Load the fine-tuned model and tokenizer from the specified directory.
    """
    print(f"Loading model from {model_dir}...")
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

def predict(model, tokenizer, codes, device):
    """
    Predict the class (flaky or not flaky) for a list of test cases.
    """
    # Tokenize the input codes
    encodings = tokenizer(
        codes,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )
    encodings = {key: val.to(device) for key, val in encodings.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)  # Get the predicted class indices

    return predictions.cpu().tolist()

def evaluate_predictions(predictions, labels):
    """
    Calculate and display evaluation metrics.
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    report = classification_report(labels, predictions, target_names=["Not Flaky", "Flaky"])
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(report)

if __name__ == "__main__":
    # Define the model directory
    model_dir = "./fine_tuned_unixcoder"
    
    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_dir, device)
    
    # Load the JSON data
    # training_data_file = 'codeBertTesting_v1.jsonl'
    training_data_file = 'codeBertTraining_v2.jsonl'
    print('\n' * 2)
    print('Loading Data...\n')

    # Read the JSONL file and parse into a list of dictionaries
    data = []
    with open(training_data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"Loaded {len(data)} samples.")

    # Extract codes and labels
    codes = [item["code"] for item in data]
    true_labels = [item["od"] for item in data]

    # Predict flaky/not flaky
    print("Predicting...")
    predictions = predict(model, tokenizer, codes, device)

    # Evaluate predictions
    evaluate_predictions(predictions, true_labels)

    # Optionally, display predictions vs. ground truth for each test case
    validate = False
    if validate:
        print("\nPredictions vs. Ground Truth:")
        for i, (code, true_label, pred_label) in enumerate(zip(codes, true_labels, predictions)):
            print(f"Test Case {i + 1}:")
            print(f"Code: {code}")
            print(f"Ground Truth: {'Flaky' if true_label == 1 else 'Not Flaky'}")
            print(f"Prediction: {'Flaky' if pred_label == 1 else 'Not Flaky'}")
            print("-" * 50)
