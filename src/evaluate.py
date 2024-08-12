from keras.src.saving import load_model

from data_loader import load_images_and_labels

# Load data
X_test, y_test = load_images_and_labels('../data/test/fire', '../data/test/non_fire', '../data/test/labels')

# Normalize pixel values
X_test = X_test / 255.0

# Load model
model = load_model('../models/fire_smoke_detection_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')
