from flask import Flask, render_template, redirect, url_for, request
import os
import cv2
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
MODEL_PATH = "pest_model.pkl"
model, class_names = joblib.load(MODEL_PATH)

# Define image size
IMG_SIZE = (128, 128)

# Define pest-to-pesticide mapping with effects on soil
PESTICIDE_MAP = {
    "Green Leafhopper": {
        "pesticides": [
            {"name": "Imidacloprid", "price": 10.0}
        ],
        "organic": "Neem Oil",
        "effect": "Highly toxic to soil microbes, reduces beneficial bacteria and fungi, affecting soil fertility."
    },
    "Planthopper": {
        "pesticides": [
            {"name": "Buprofezin", "price": 15.0}
        ],
        "organic": "Beauveria bassiana",
        "effect": "Minimal direct impact on soil but reduces beneficial soil arthropods, affecting soil aeration."
    },
    "Rice Bug": {
        "pesticides": [
            {"name": "Lambda-cyhalothrin", "price": 20.0}
        ],
        "organic": "Pyrethrin",
        "effect": "Binds to soil, toxic to earthworms, disrupts soil food web, lowers organic matter decomposition."
    },
    "Rice Leaf Roller": {
        "pesticides": [
            {"name": "Chlorpyrifos", "price": 25.0}
        ],
        "organic": "Bacillus thuringiensis",
        "effect": "Persistent in soil for over a year, disrupts soil enzymes, kills beneficial nematodes, can leach into groundwater."
    },
    "Rice Stem Borer": {
        "pesticides": [
            {"name": "Fipronil", "price": 30.0}
        ],
        "organic": "Trichogramma",
        "effect": "Long-term soil toxicity, reduces microbial balance and earthworms, alters nitrogen cycle."
    },
}

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Your home page template

# Register route


def predict_pest(image_path):
    """Predict the pest and recommend multiple pesticides sorted by cost."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE)  # Resize to the correct size
    img = img / 255.0  # Normalize pixel values to range [0, 1]

    # Ensure image has the right dimensions for prediction
    img = img.flatten().reshape(1, -1)  # Flatten the image and reshape to 2D array

    pred_label = model.predict(img)[0]
    pest_name = class_names[pred_label].strip().title()

    # Debugging: Print pest name and predictions
    print(f"Detected Pest: {pest_name}")

    # Fetch pesticides, organic solutions, and effects
    pesticide_info = PESTICIDE_MAP.get(pest_name, {"pesticides": [], "organic": "No organic alternatives", "effect": "No soil impact information available"})

    # Sort pesticides by price (ascending)
    sorted_pesticides = sorted(pesticide_info["pesticides"], key=lambda x: x["price"])

    # Print the debug info
    print(f"Sorted Pesticides: {sorted_pesticides}")
    print(f"Organic Alternatives: {pesticide_info['organic']}")
    print(f"Soil Effect: {pesticide_info['effect']}")

    return pest_name, sorted_pesticides, pesticide_info["organic"], pesticide_info["effect"]

# Pest detection route
@app.route("/pestdetection", methods=["GET", "POST"])
def pestdetection():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("pestdetection.html", error="No file uploaded")

        file = request.files["file"]
        if file.filename == "":
            return render_template("pestdetection.html", error="No file selected")

        # Save file
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Get prediction
        pest_name, pesticides, organic, effect = predict_pest(file_path)

        return render_template("pestdetection.html", filename=file.filename, pest=pest_name, pesticides=pesticides, effect=effect, organic=organic)

    return render_template("pestdetection.html")

if __name__ == "__main__":
    app.run(debug=True)