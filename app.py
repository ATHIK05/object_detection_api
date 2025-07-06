import os

# Download model from Google Drive if not present
try:
    import gdown
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "gdown"])
    import gdown

drive_url = "https://drive.google.com/drive/folders/1MAxozJJ0AxKIxHBwSn9loaVqh6j_5uPs?usp=sharing"
model_dir = "model_dir"
if not os.path.exists(model_dir):
    gdown.download_folder(drive_url, output=model_dir, quiet=False, use_cookies=False)

from flask import Flask, request, jsonify
from tf_model_loader import load_model
from utils import preprocess_image, extract_labels

app = Flask(__name__)
detector = load_model()

@app.route("/", methods=["GET"])
def home():
    return "🟢 Object Detector API is running"

@app.route("/detect", methods=["POST"])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    img_tensor, _ = preprocess_image(image_file)
    result = detector(img_tensor)
    label_counts = extract_labels(result)
    
    return jsonify({
        "status": 200,
        "count": sum(label_counts.values()),
        "labels": label_counts
    })

if __name__ == "__main__":
    app.run(debug=True)
