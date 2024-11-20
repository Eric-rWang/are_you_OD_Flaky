import json
from transformers import RobertaTokenizer

# Load the JSON data
training_data_file = 'codeBertTraining_v2.jsonl'
print('\n'*2)
print('Data Is Loaded')

data = []
with open(training_data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Convert data into a format suitable for training
texts = [item["code"] for item in data]
labels = [item["od"] for item in data]

# Load CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Tokenize the data
tokenized_texts = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)

print('Finished Tokenizing\n')

import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx])
        }

# Create dataset
dataset = TestDataset(tokenized_texts, labels)

print('Finished Creating Dataset\n')

from sklearn.model_selection import train_test_split

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize the split data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Create datasets
train_dataset = TestDataset(train_encodings, train_labels)
val_dataset = TestDataset(val_encodings, val_labels)

print('Dataset Split\n')

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments

# Load CodeBERT model for binary classification
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

print('Done Fine-Tuning\n')

# Evaluate on validation set
results = trainer.evaluate()
print(results)

# Predict on new data
test_code = ["def test_new_case(): ..."]
test_encodings = tokenizer(test_code, truncation=True, padding=True, max_length=512, return_tensors="pt")
predictions = model(**test_encodings)
print(predictions.logits)  # Logits indicate the confidence for each class

print('Processing Complete\n')