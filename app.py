import os
import cv2
import numpy as np
import joblib
from flask import Flask, request, render_template

# Load trained model
MODEL_PATH = "pest_model.pkl"
model, class_names = joblib.load(MODEL_PATH)

# Define image size
IMG_SIZE = (128, 128)

# Define pest-to-pesticide mapping with effects on soil
PESTICIDE_MAP = {
    "Green Leafhopper": {
        "pesticide": "Imidacloprid",
        "effect": "Highly toxic to soil microbes, reduces beneficial bacteria and fungi, affecting soil fertility."
    },
    "Planthopper": {
        "pesticide": "Buprofezin",
        "effect": "Minimal direct impact on soil but reduces beneficial soil arthropods, affecting soil aeration."
    },
    "Rice Bug": {
        "pesticide": "Lambda-cyhalothrin",
        "effect": "Binds to soil, toxic to earthworms, disrupts soil food web, lowers organic matter decomposition."
    },
    "Rice Leaf Roller": {
        "pesticide": "Chlorpyrifos",
        "effect": "Persistent in soil for over a year, disrupts soil enzymes, kills beneficial nematodes, can leach into groundwater."
    },
    "Rice Stem Borer": {
        "pesticide": "Fipronil",
        "effect": "Long-term soil toxicity, reduces microbial balance and earthworms, alters nitrogen cycle."
    },
}

# Create Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_pest(image_path):
    """Predict the pest and recommend a pesticide with soil effects."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE).flatten() / 255.0
    img = img.reshape(1, -1)

    pred_label = model.predict(img)[0]
    pest_name = class_names[pred_label].strip().title()  # Normalize name formatting

    # Debugging: Print pest name to check its format
    print(f"Detected Pest: {pest_name}")

    # Get pesticide recommendation and its effect
    pesticide_info = PESTICIDE_MAP.get(pest_name, {"pesticide": "No recommendation available", "effect": "No soil impact information available"})
    pesticide = pesticide_info["pesticide"]
    effect = pesticide_info["effect"]

    return pest_name, pesticide, effect

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")

        # Save file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Get prediction
        pest_name, pesticide, effect = predict_pest(file_path)

        return render_template("index.html", filename=file.filename, pest=pest_name, pesticide=pesticide, effect=effect)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
