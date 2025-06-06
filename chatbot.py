from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})  # Enable CORS for frontend

# Load API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found! Make sure it's set in the environment variables.")

# Define API URL
MODEL_NAME = "gemini-1.5-pro-002"  # ✅ Use a valid model name
url = f"https://generativelanguage.googleapis.com/v1/models/{MODEL_NAME}:generateContent?key={api_key}"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json.get("message")
        if not user_input:
            return jsonify({"error": "Message is required"}), 400
        
        # Prepare API request
        headers = {"Content-Type": "application/json"}
        data = {"contents": [{"parts": [{"text": user_input}]}]}
        
        # Make API call
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()

        # Extract response text
        bot_response = response_json.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: No response from AI.")
        
        return jsonify({"response": bot_response})

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
