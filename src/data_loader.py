# src/data_loader.py

import os
import cv2
import numpy as np


def load_images_and_labels(fire_folder, non_fire_folder, label_folder, img_size=(224, 224)):
    images = []
    labels = []

    def extract_label(label_line):
        parts = label_line.strip().split()
        if len(parts) > 0:
            return int(parts[0])
        return None

    # Helper function to load images and labels from a directory
    def load_from_folder(image_folder, label_folder, images, labels):
        for filename in sorted(os.listdir(image_folder)):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(image_folder, filename)
                label_filename = os.path.splitext(filename)[0] + ".txt"
                label_path = os.path.join(label_folder, label_filename)

                if os.path.exists(label_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, img_size)
                        images.append(img)

                        with open(label_path, 'r') as file:
                            label_line = file.readline()
                            label = extract_label(label_line)
                            if label is not None:
                                labels.append(label)
                            else:
                                # Remove the last added image if the label is not valid
                                images.pop()
                else:
                    print(f"Label file {label_path} does not exist.")

    # Load fire images and labels
    load_from_folder(fire_folder, label_folder, images, labels)

    # Load non_fire images and labels
    load_from_folder(non_fire_folder, label_folder, images, labels)

    return np.array(images), np.array(labels)
