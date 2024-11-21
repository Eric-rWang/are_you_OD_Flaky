import json
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Load the JSON data
training_data_file = 'codeBertTraining_v2.jsonl'
print('\n' * 2)
print('Loading Data...\n')

# Read the JSONL file and parse into a list of dictionaries
data = []
with open(training_data_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} samples.")

# Extract text and labels
texts = [item["code"] for item in data]
labels = [item["od"] for item in data]

for i, (code, label) in enumerate(zip(texts, labels)):
    if not isinstance(label, int) or label not in [0, 1]:
        print(f"Invalid label at index {i}: {label}")
    if len(code.strip()) == 0:
        print(f"Empty code at index {i}")

# Check for invalid or empty samples
empty_texts = [i for i, text in enumerate(texts) if len(text.strip()) == 0]
if empty_texts:
    print(f"Warning: Empty test cases found at indices: {empty_texts}")

# Load CodeBERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Tokenize the data with truncation
print('Tokenizing Data...\n')
tokenized_texts = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=512,
    return_tensors="pt"
)

print('Finished Tokenizing...\n')

# Custom Dataset Class
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

# Check sequence lengths
max_length = 512
lengths = [len(tokenizer.encode(text)) for text in texts]
print(f"Longest sequence length: {max(lengths)}")
if max(lengths) > max_length:
    print(f"Warning: {sum(l > max_length for l in lengths)} sequences were truncated to {max_length} tokens.")

# Split into train and validation sets
print('Splitting Dataset...\n')
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize split data
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Create datasets
train_dataset = TestDataset(train_encodings, train_labels)
val_dataset = TestDataset(val_encodings, val_labels)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}\n")

# Initialize the CodeBERT model
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=8,  # Reduce batch size
    per_device_eval_batch_size=8,
    warmup_steps=100,  # Gradual warmup
    weight_decay=0.01,
    learning_rate=1e-5,  # Lower learning rate
    max_grad_norm=1.0,  # Enable gradient clipping
    logging_dir='./logs',
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
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
model_dir = "./fine_tuned_codebert"
if os.path.exists(model_dir):
    print(f"Loading the fine-tuned model from {model_dir}")
    model = RobertaForSequenceClassification.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
else:
    print("No fine-tuned model found. Training a new model...")
    # Add your training code here
    print('Starting Training...\n')
    trainer.train()
    print('Training Complete!\n')
    print(f"Saving the fine-tuned model to {model_dir}")
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)

# Evaluate on validation set
print('Evaluating on Validation Set...\n')
results = trainer.evaluate()
print(f"Evaluation Results: {results}")

# Check if MPS is available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to MPS device
model.to(device)

# Tokenize the new data
test_code = ["def test_new_case(): pass"]
test_encodings = tokenizer(test_code, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Move inputs to MPS device
test_encodings = {key: val.to(device) for key, val in test_encodings.items()}

# Predict
with torch.no_grad():
    predictions = model(**test_encodings)
print(f"Logits: {predictions.logits}")

