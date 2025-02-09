from flask import Flask, request, render_template
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Initialize Flask app
app = Flask(__name__)

# Load trained model
classifier = joblib.load("cat_vs_dog_classifier.pkl")  # Ensure the model is saved

# Define image size
IMG_SIZE = (64, 64)

# Function to process uploaded image
def process_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)

    # Extract HOG features
    hog_features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), block_norm='L2-Hys')

    return [hog_features]

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        file_path = "static/uploaded_image.jpg"  # Save file temporarily
        file.save(file_path)

        # Process image & predict
        features = process_image(file_path)
        prediction = classifier.predict(features)
        label = "Dog" if prediction[0] == 1 else "Cat"

        return render_template("index.html", label=label, image=file_path)

    return render_template("index.html", label=None)

if __name__ == "__main__":
    app.run(debug=True)
