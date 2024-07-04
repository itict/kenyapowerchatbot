import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import joblib

# Get the parent directory and add it to the sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from preprocessing import KPLCDataset
from models.kplc_chatbot_model import KPLCChatbotModel

# Check and create the directory for models if it doesn't exist
vectorizer_path = os.path.join(parent_dir, 'backend', 'models', 'tfidf_vectorizer.pkl')
os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

print(f"Current working directory: {os.getcwd()}")  # Print current working directory for debugging

# Load dataset and vectorizer
try:
    vectorizer = joblib.load(vectorizer_path)
except FileNotFoundError as e:
    print(f"Error: Could not find vectorizer file at {vectorizer_path}.")
    print(f"Make sure the vectorizer file exists at the specified location.")
    sys.exit(1)  # Exit the script if vectorizer file is not found

dataset_path = os.path.join(parent_dir, 'data', 'kplc_data.json')
dataset = KPLCDataset(dataset_path, vectorizer)

# Read the number of classes from the JSON file
with open(dataset_path, 'r', encoding='utf-8') as file:
    intents = json.load(file)
num_classes = len(intents['intents'])  # Number of unique classes

# Split dataset into train and test sets
train_indices, test_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Parameters
input_size = len(train_dataset[0][0])  # Size of TF-IDF vector
hidden_size = 128

# Define the model
model = KPLCChatbotModel(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Train the model
num_epochs = 10
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (texts, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(texts)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# Save the trained model
model_path = os.path.join(parent_dir, 'backend', 'models', 'kplc_chatbot_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved successfully at: {model_path}")
