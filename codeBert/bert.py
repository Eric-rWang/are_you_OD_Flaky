import torch
import os
import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
from sklearn.preprocessing import LabelEncoder

# Load Tokenizer and Model
# Assuming we're using a RoBERTa-based model, since CodeBERT uses a RoBERTa architecture.
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load Pre-trained CodeBERT for Classification Task
classification_model = RobertaForSequenceClassification.from_pretrained(
    "microsoft/codebert-base",
    num_labels=2  # Assuming 3 labels: not flaky, brittle, and victim (for now 2, brittle and victim)
)

# Load training data from JSON file
training_data_file = 'codeBertTraining.jsonl'

data = []
with open(training_data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
with open(training_data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Prepare dataset for training CodeBERT

def prepare_dataset(data):
    """
    Prepare dataset for training the classification model.
    :param data: List of dictionaries with keys: 'code', 'label', 'fix'.
    :return: Dataset object for training.
    """
    codes = [example['code'] for example in data]
    label_strings = [example['label'] for example in data]

    # Encode string labels to integers
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(label_strings)

    if not codes:
        raise ValueError("No code samples provided for tokenization.")

    inputs = tokenizer(codes, padding=True, truncation=True, return_tensors="pt")
    print(f"Inputs shape: {inputs['input_ids'].shape}")
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    inputs['labels'] = labels_tensor
    print(f"Labels shape: {inputs['labels'].shape}")

    return Dataset.from_dict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels_tensor
    })

train_dataset = prepare_dataset(train_data)
val_dataset = prepare_dataset(val_data)
dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
})

# Prepare Dataset for Sequence to Sequence Training

def prepare_seq2seq_dataset(data):
    """
    Prepare dataset for fine-tuning the sequence generation model.
    :param data: List of dictionaries with keys: 'code' and 'fix'.
    :return: Dataset object for training.
    """
    codes = [example['code'] for example in data]
    fixes = [example['fix'] for example in data]

    if not codes or not fixes:
        raise ValueError("No code or fix samples provided for tokenization.")

    # Tokenize inputs
    inputs = tokenizer(codes, padding=True, truncation=True, return_tensors="pt")
    print(f"Inputs shape: {inputs['input_ids'].shape}")

    # Tokenize labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(fixes, padding=True, truncation=True, return_tensors="pt")['input_ids']
    
    # Replace padding tokens in labels with -100 so that they are ignored in the loss calculation
    labels[labels == tokenizer.pad_token_id] = -100

    print(f"Labels shape: {labels.shape}")

    return Dataset.from_dict({
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': labels
    })

seq2seq_dataset = DatasetDict({
    "train": prepare_seq2seq_dataset(data),
    # Add validation and test datasets as needed
})

# Fine-tuning CodeBERT for Classification
classification_training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,  # To avoid needing additional dependencies
)

from transformers import Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

classification_trainer = Trainer(
    model=classification_model,
    args=classification_training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    # validation and evaluation can be added here
)

# Train the Model
classification_trainer.train()

fixing_model = RobertaForCausalLM.from_pretrained("microsoft/codebert-base")

# Fixing Flaky Tests - Sequence to Sequence Generation
# Using CodeBERT for Fixing Flaky Tests
fixing_training_args = TrainingArguments(
    output_dir="./results_fixing",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_steps=10,
    push_to_hub=False,
)

# Define data collator
# from transformers import DataCollatorForLanguageModeling

# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# seq2seq_trainer = Trainer(
#     model=fixing_model,
#     args=fixing_training_args,
#     train_dataset=seq2seq_dataset["train"],
#     data_collator=data_collator,
# )

# Train the Model for Fixing Flaky Tests
# seq2seq_trainer.train()

# Evaluation
# Define evaluation functions to test accuracy of classification and correctness of fixes.
