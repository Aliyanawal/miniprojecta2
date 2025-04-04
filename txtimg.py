from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import torch
from diffusers import StableDiffusionPipeline
import os

app = Flask(__name__, static_folder="static")
CORS(app)

os.makedirs("static", exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "runwayml/stable-diffusion-v1-5"

# Load model with optimizations
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe.to(device)
pipe.enable_attention_slicing()  # ✅ Reduce memory usage

@app.route("/generate", methods=["POST"])
def generate_image():
    try:
        data = request.get_json()
        prompt = data.get("prompt", "").strip()
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Generate image (Lower resolution, fewer steps)
        image = pipe(prompt, height=384, width=384, num_inference_steps=10).images[0]

        image_filename = "generated_image.png"
        image_path = os.path.join("static", image_filename)
        image.save(image_path)

        image_url = url_for('serve_static', filename=image_filename, _external=True)
        return jsonify({"image_url": image_url})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
