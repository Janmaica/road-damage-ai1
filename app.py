from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# -------------------------------
# Load YOLO model (deployment-safe)
# -------------------------------
MODEL_PATH = "best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("best.pt not found in project directory")

model = YOLO(MODEL_PATH)


# -------------------------------
# Health check route
# -------------------------------
@app.route("/")
def home():
    return "Road Damage Detection API is running"


# -------------------------------
# Prediction route
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Convert image to OpenCV format
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Run YOLO prediction
    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0])
            })

    return jsonify({
        "detections": detections
    })


# -------------------------------
# Local development only
# (Render / Railway will ignore this)
# -------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)