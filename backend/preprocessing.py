import re
import string
import json
import joblib
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Preprocess text function (adjust as necessary)
def preprocess_text(text):
    # Example preprocessing steps
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text

class KPLCDataset(Dataset):
    def __init__(self, json_file, vectorizer):
        self.vectorizer = vectorizer

        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['intents']  # Assuming the JSON structure

        self.texts = [pattern for intent in self.data for pattern in intent['patterns']]
        self.labels = [intent['tag'] for intent in self.data for _ in intent['patterns']]

        # Fit vectorizer on the texts
        self.vectorizer.fit(self.texts)

        # Encode labels to integers
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.encoded_labels[idx]
        text_vectorized = self.vectorizer.transform([text]).toarray()[0]
        return torch.FloatTensor(text_vectorized), torch.tensor(label, dtype=torch.long)

# Example usage:
if __name__ == "__main__":
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)

    # Create dataset instance
    dataset = KPLCDataset('../data/kplc_data.json', vectorizer)

    # Save vectorizer using joblib
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
