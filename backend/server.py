from flask import Flask, request, jsonify
import torch
import joblib
import json
import random
from models.kplc_chatbot_model import KPLCChatbotModel
from preprocessing import preprocess_text
from flask_cors import CORS

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# File paths
model_file = 'models/kplc_chatbot_model.pth'
vectorizer_file = 'models/tfidf_vectorizer.pkl'
intents_file = '../data/kplc_data.json'  # Update this to your actual JSON file path

# Load the vectorizer and intents JSON file
vectorizer = joblib.load(vectorizer_file)
with open(intents_file, 'r', encoding='utf-8') as file:
    intents = json.load(file)

# Define model parameters (adjust based on your model)
input_size = 216  # Adjust based on your TF-IDF vector size
hidden_size = 128  # Adjust as per your model design
num_classes = len(intents['intents'])  # Number of classes based on the intents

# Initialize the model
model = KPLCChatbotModel(input_size, hidden_size, num_classes)

# Load pretrained model state_dict
checkpoint = torch.load(model_file, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)

model.eval()

# Inference function
def predict_text(text):
    try:
        # Preprocess the text
        preprocessed_text = preprocess_text(text)  # Implement preprocess_text as required
        input_vector = vectorizer.transform([preprocessed_text]).toarray()
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output.data, 1)

        return predicted.item()

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# Map class index to tag
def class_to_tag(class_index):
    try:
        tags = [intent['tag'] for intent in intents['intents']]
        return tags[class_index]
    except IndexError:
        print(f"Invalid class index: {class_index}")
        return None

# Get response for tag
def get_response_for_tag(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that."

# Route for chatbot interaction
@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    try:
        # Get input data from request
        data = request.get_json()
        user_message = data['message']

        # Perform prediction
        predicted_class = predict_text(user_message)

        if predicted_class is not None:
            # Map class to tag and get response
            predicted_tag = class_to_tag(predicted_class)
            if predicted_tag:
                bot_response = get_response_for_tag(predicted_tag)
                return jsonify({'response': bot_response}), 200
            else:
                return jsonify({'response': 'Failed to map the class to a tag.'}), 500
        else:
            return jsonify({'response': 'Failed to process the request.'}), 500

    except KeyError:
        return jsonify({'error': 'Invalid request format. Ensure you send JSON with key "message".'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
