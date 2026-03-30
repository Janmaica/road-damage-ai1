from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load your trained YOLO model
model = YOLO("best.pt")

@app.route("/")
def home():
    return "Road Damage Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]

    # Convert image to OpenCV format
    img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    # Run YOLO prediction
    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0])
            })

    return jsonify({"detections": detections})

if __name__ == "__main__":
    # 🔥 IMPORTANT FIX HERE
    app.run(host="0.0.0.0", port=5000, debug=True)