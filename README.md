# Cat vs Dog Classifier (ML + Flask Web App)

## Project Overview
This project is a **Machine Learning model** that classifies images as either **cats or dogs**.
It includes a **Flask web app** where users can **upload an image** and get an instant prediction.

## Technologies Used
- Python
- Flask (for the web app)
- OpenCV (for image processing)
- Scikit-learn (for machine learning)
- Histogram of Oriented Gradients (HOG) (for feature extraction)

## Installation & Setup

### 1. Clone the Repository
```
git clone https://github.com/rnaltami/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier
```

### 2. Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Run the Flask Web App
```
python app.py
```
Then open `http://127.0.0.1:5000/` in your browser.

## How to Use the Web App
1. Upload an image (of a cat or dog).
2. The model analyzes the image and predicts if it’s a Cat or Dog.
3. The result is displayed along with the uploaded image.

## Training the Model
If you want to retrain the model or tweak it, run:
```
python classifier.py
```
This will:
- Load and preprocess the dataset
- Train a Random Forest classifier
- Save the trained model as `cat_vs_dog_classifier.pkl`

## Future Improvements
- Improve accuracy using Convolutional Neural Networks (CNNs).
- Deploy the web app online using Render, Heroku, or AWS.
- Allow batch uploads for multiple images.

## Troubleshooting

### 1. Flask App Doesn't Start?
Make sure you activated the virtual environment:
```
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 2. Model Not Found?
If you get an error saying `"cat_vs_dog_classifier.pkl" not found`, retrain the model:
```
python classifier.py
```
Then restart Flask.

### 3. Can't Push to GitHub?
Try forcing the push:
```
git push --force origin main
```

## Contribute
If you’d like to improve the project, feel free to fork the repo and submit a **pull request**.

For questions, contact me via [GitHub Issues](https://github.com/rnaltami/cat-vs-dog-classifier/issues).

