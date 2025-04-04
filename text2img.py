import requests
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load API key securely
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

# MiniMax API endpoint (Replace with actual URL from MiniMax docs)
MINIMAX_API_URL = "https://api.segmind.com/v1/vision/sd1.5-txt2img"  # ✅ Correct Segmind URL


app = Flask(__name__)
CORS(app)  # Allow frontend requests

def generate_image(prompt):
    """Send request to MiniMax API to generate an image from text."""
    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "prompt": prompt,
        "model": "minimax-image-gen-v1",  # Adjust model name if needed
        "size": "1024x1024"
    }

    response = requests.post(MINIMAX_API_URL, json=data, headers=headers)

    if response.status_code == 200:
        return response.json().get("image_url")  # Assuming API returns an image URL
    else:
        return None

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    image_url = generate_image(prompt)

    if image_url:
        return jsonify({"image_url": image_url})
    else:
        return jsonify({"error": "Image generation failed"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5002)

