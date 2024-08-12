import cv2
import numpy as np
import os

from keras.src.saving import load_model

# Load model
model_path = '../models/fire_smoke_detection_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at '{model_path}'")

model = load_model(model_path)


def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image at path '{image_path}' could not be found.")
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def predict(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction[0][0]


if __name__ == "__main__":
    # Debugging information
    print(f"Current working directory: {os.getcwd()}")
    print(f"Contents of the data directory: {os.listdir('../data')}")

    # Update the image path to the generated image in the data directory
    image_path = '../data/test_image.jpg'  # Ensure this path is correct
    try:
        prediction = predict(image_path)
        print(f'Prediction: {"Fire/Smoke Detected" if prediction > 0.5 else "No Fire/Smoke Detected"}')
    except FileNotFoundError as e:
        print(e)
