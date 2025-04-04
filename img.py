import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import os

# Load API key from environment variable
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests

def generate_image_description(image):
    """Uses Gemini Pro Vision to generate a description for an image."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([image, "Describe this image in detail."])
    return response.text

@app.route("/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read the image
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    # Generate description
    description = generate_image_description(image)

    return jsonify({"description": description})

if __name__ == "__main__":
    app.run(debug=True)
