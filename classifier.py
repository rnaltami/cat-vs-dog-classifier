import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Images
def load_images_from_folder(folder, label, image_size=(64, 64)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            data.append((img, label))
    return data

# Step 2: Extract Features (To be added)
# Step 3: Train Model (To be added)
# Step 4: Evaluate Model (To be added)

if __name__ == "__main__":
    print("Cat vs Dog Classifier - Basic Structure Ready!")
