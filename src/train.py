import os

from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator

from data_loader import load_images_and_labels
from model import create_model

# Verify directories exist
required_dirs = [
    '../data/train/fire',
    '../data/train/non_fire',
    '../data/train/labels',
    '../data/val/fire',
    '../data/val/non_fire',
    '../data/val/labels'
]

for directory in required_dirs:
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Required directory not found: {directory}")

# Load data
X_train, y_train = load_images_and_labels('../data/train/fire', '../data/train/non_fire', '../data/train/labels')
X_val, y_val = load_images_and_labels('../data/val/fire', '../data/val/non_fire', '../data/val/labels')

# Check if lengths match
print(f"Training data: {X_train.shape}, {y_train.shape}")
print(f"Validation data: {X_val.shape}, {y_val.shape}")

if len(X_train) != len(y_train):
    raise ValueError(f"Training data length mismatch: {len(X_train)} images vs {len(y_train)} labels")
if len(X_val) != len(y_val):
    raise ValueError(f"Validation data length mismatch: {len(X_val)} images vs {len(y_val)} labels")

# Normalize pixel values
X_train = X_train / 255.0
X_val = X_val / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow(X_train, y_train, batch_size=32)
val_generator = datagen.flow(X_val, y_val, batch_size=32)

# Create model
model = create_model()

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the model
model.fit(train_generator, epochs=30, validation_data=val_generator, callbacks=[early_stopping, reduce_lr])

# Save the model
model.save('../models/fire_smoke_detection_model.h5')
