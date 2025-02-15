import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Define dataset path
DATASET_PATH = "C:/Users/KeshavSharma/Desktop/pesticides"  # Change to your dataset path

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

def load_dataset():
    """Load images and labels from dataset."""
    images, labels = [], []
    class_names = os.listdir(DATASET_PATH)

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)  # Resize image
            img = img.flatten() / 255.0  # Normalize
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels), class_names

# Load dataset
print("Loading dataset...")
X, y, class_names = load_dataset()
print(f"Dataset loaded: {len(X)} images, {len(set(y))} classes.")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
print("Training SVM model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
print("Model training completed.")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump((model, class_names), "pest_model.pkl")
print("Model saved as pest_model.pkl")

def predict_pest(image_path):
    """Predict the pest and recommend a pesticide with soil effects."""
    model, class_names = joblib.load("pest_model.pkl")
    img = cv2.imread(image_path)
    img = cv2.resize(img, IMG_SIZE).flatten() / 255.0
    img = img.reshape(1, -1)

    pred_label = model.predict(img)[0]
    pest_name = class_names[pred_label]
    
    # Get pesticide recommendation and its effect
    if pest_name in PESTICIDE_MAP:
        pesticide_info = PESTICIDE_MAP[pest_name]
        pesticide = pesticide_info["pesticide"]
        effect = pesticide_info["effect"]
    else:
        pesticide = "No recommendation available"
        effect = "No soil impact information available"

    return pest_name, pesticide, effect

# Example Prediction
test_img = "C:/Users/KeshavSharma/Desktop/pesticides/Green Leafhopper/sample.jpg"  # Change to your image path
if os.path.exists(test_img):
    pest, pesticide, effect = predict_pest(test_img)
    print(f"Pest Detected: {pest}")
    print(f"Recommended Pesticide: {pesticide}")
    print(f"Effect on Soil: {effect}")
