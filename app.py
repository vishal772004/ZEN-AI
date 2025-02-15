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
        "pesticides": [
            {"name": "Imidacloprid", "price": 500},
            {"name": "Thiamethoxam", "price": 450},
            {"name": "Acetamiprid", "price": 430}
        ],
        "organic": "Neem Oil, Spinosad",
        "effect": "Highly toxic to soil microbes, reduces beneficial bacteria and fungi, affecting soil fertility."
    },
    "Planthopper": {
        "pesticides": [
            {"name": "Buprofezin", "price": 600},
            {"name": "Pymetrozine", "price": 550},
            {"name": "Dinotefuran", "price": 530}
        ],
        "organic": "Beauveria bassiana (fungal biopesticide), Neem Extract",
        "effect": "Minimal direct impact on soil but reduces beneficial soil arthropods, affecting soil aeration."
    },
    "Rice Bug": {
        "pesticides": [
            {"name": "Lambda-cyhalothrin", "price": 700},
            {"name": "Deltamethrin", "price": 650},
            {"name": "Fenvalerate", "price": 620}
        ],
        "organic": "Garlic extract, Chrysanthemum flower extract",
        "effect": "Binds to soil, toxic to earthworms, disrupts soil food web, lowers organic matter decomposition."
    },
    "Rice Leaf Roller": {
        "pesticides": [
            {"name": "Chlorpyrifos", "price": 750},
            {"name": "Carbaryl", "price": 720},
            {"name": "Quinalphos", "price": 710}
        ],
        "organic": "Bt (Bacillus thuringiensis), Azadirachtin",
        "effect": "Persistent in soil for over a year, disrupts soil enzymes, kills beneficial nematodes, can leach into groundwater."
    },
    "Rice Stem Borer": {
        "pesticides": [
            {"name": "Fipronil", "price": 900},
            {"name": "Chlorantraniliprole", "price": 850},
            {"name": "Emamectin Benzoate", "price": 820}
        ],
        "organic": "Trichogramma wasps, Neem Cake",
        "effect": "Long-term soil toxicity, reduces microbial balance and earthworms, alters nitrogen cycle."
    },
    "Aphids": {
        "pesticides": [
            {"name": "Malathion", "price": 550},
            {"name": "Flonicamid", "price": 520},
            {"name": "Pymetrozine", "price": 500}
        ],
        "organic": "Ladybugs, Neem Oil",
        "effect": "Can affect beneficial pollinators if sprayed widely."
    },
    "Whiteflies": {
        "pesticides": [
            {"name": "Spiromesifen", "price": 630},
            {"name": "Buprofezin", "price": 600},
            {"name": "Pyriproxyfen", "price": 580}
        ],
        "organic": "Sticky traps, Garlic Spray",
        "effect": "Residues persist in soil for months, affecting soil-dwelling insects."
    },
}

# Create Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_pest(image_path):
    """Predict the pest and recommend multiple pesticides sorted by cost."""
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE).flatten() / 255.0
    img = img.reshape(1, -1)

    pred_label = model.predict(img)[0]
    pest_name = class_names[pred_label].strip().title()

    # Debugging: Print pest name
    print(f"Detected Pest: {pest_name}")

    # Fetch pesticides, organic solutions, and effects
    pesticide_info = PESTICIDE_MAP.get(pest_name, {"pesticides": [], "organic": "No organic alternatives", "effect": "No soil impact information available"})

    # Sort pesticides by price (ascending)
    sorted_pesticides = sorted(pesticide_info["pesticides"], key=lambda x: x["price"])

    return pest_name, sorted_pesticides, pesticide_info["organic"], pesticide_info["effect"]

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
        pest_name, pesticides, organic, effect = predict_pest(file_path)

        return render_template("index.html", filename=file.filename, pest=pest_name, pesticides=pesticides, organic=organic, effect=effect)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
