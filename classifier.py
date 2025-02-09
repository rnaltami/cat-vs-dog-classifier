import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage.feature import hog



# Define dataset path
DATASET_PATH = "dataset"

# Define image size for processing
IMG_SIZE = (64, 64)

# Function to load a limited number of images
def load_images_and_labels(dataset_path, limit=500):  # Limit to 500 per class
    images = []
    labels = []
    
    for label, category in enumerate(["cats", "dogs"]):  # 0 for cats, 1 for dogs
        folder_path = os.path.join(dataset_path, category)
        img_names = os.listdir(folder_path)[:limit]  # Select only first `limit` images
        
        for img_name in img_names:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            if img is not None:
                img = cv2.resize(img, IMG_SIZE)  # Resize to (64, 64)
                
                # 🔹 Extract HOG features instead of raw pixels
                hog_features = hog(img, orientations=8, pixels_per_cell=(16, 16),
                                   cells_per_block=(2, 2), block_norm='L2-Hys')
                
                images.append(hog_features)  # Use HOG features instead of flattened image
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Reload dataset using improved features
X, y = load_images_and_labels(DATASET_PATH, limit=500)


# Print dataset size
print(f"Dataset Loaded: {X.shape[0]} total images")

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset split info
print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

from sklearn.svm import SVC

# Train a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Print results
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.4f}")

import joblib

import joblib

# Save the trained model
joblib.dump(classifier, "cat_vs_dog_classifier.pkl")

print("Model saved as cat_vs_dog_classifier.pkl")

