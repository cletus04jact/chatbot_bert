import json
import random
import nltk
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np

# Load the intents file
with open('Agricultural(tamil).json') as file:
    intents = json.load(file)

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intents['intents']))

# Prepare the data
patterns = []
tags = []
responses = {}

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    
    # Store the responses for each tag (intent)
    responses[intent['tag']] = intent['responses']

# Encode the labels (tags) into numbers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Tokenization and Encoding
class ChatbotDataset(Dataset):
    def __init__(self, patterns, labels, tokenizer, max_len):
        self.patterns = patterns
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.patterns)

    def __getitem__(self, item):
        pattern = self.patterns[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            pattern,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Parameters
MAX_LEN = 50
BATCH_SIZE = 16

# Create DataLoader for training
train_data = ChatbotDataset(patterns, labels, tokenizer, MAX_LEN)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)

# Training the BERT model
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)

# Training loop
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

# Save the trained model
model.save_pretrained('chatbot_bert_model')
tokenizer.save_pretrained('chatbot_bert_model')

# Function to predict response using the trained BERT model
def chatbot_response(user_input):
    model.eval()

    # Tokenize the user input and prepare it for the model
    encoding = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict the label (intent)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get the predicted tag
    predicted_label = torch.argmax(logits, dim=1).item()

    # Get the tag name and return a random response from the corresponding intent
    tag = label_encoder.inverse_transform([predicted_label])[0]
    return random.choice(responses[tag])

# Example usage
print("Chatbot: Hello! Ask me anything.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print(f"Chatbot: {response}")
