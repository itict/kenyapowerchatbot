import json
import os  # Importing os module for file operations
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load your data
try:
    with open('data/kplc_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error loading JSON file: {e}")
    data = {}

# Print the type and structure of loaded data
print("Loaded data type:", type(data))
print("Data contents:", data)

# Check if data contains the 'intents' key and it is a list
if 'intents' in data and isinstance(data['intents'], list):
    print("Intents found in the JSON data.")
    
    # Initialize a list to store all patterns
    all_patterns = []

    # Iterate over each intent and extract patterns
    for intent in data['intents']:
        if 'patterns' in intent and isinstance(intent['patterns'], list):
            patterns = intent['patterns']
            all_patterns.extend(patterns)
            print(f"Extracted patterns for tag '{intent['tag']}': {patterns}")
        else:
            print(f"No patterns found for tag '{intent['tag']}' or patterns are not in expected format.")
    
    # Print all extracted patterns
    print("All extracted patterns:", all_patterns)

    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed

    # Fit the vectorizer on the patterns
    if all_patterns:
        print("Fitting the vectorizer on the extracted patterns.")
        vectorizer.fit(all_patterns)

        # Save the fitted vectorizer
        try:
            os.makedirs('backend/models', exist_ok=True)  # Ensure the directory exists
            joblib.dump(vectorizer, 'backend/models/tfidf_vectorizer.pkl')
            print("Vectorizer has been fitted and saved successfully.")
        except Exception as e:
            print(f"Error saving vectorizer: {e}")
    else:
        print("No patterns extracted to fit the vectorizer.")
else:
    print("Error: JSON file does not contain expected list of intents or intents are not in expected format.")
