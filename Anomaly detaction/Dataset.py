import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class UnhealthyXRayDataset(tf.keras.utils.Sequence):
    def __init__(self, image_folder, resize_shape=(256, 256)):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder) if
                            file.endswith(('.jpeg', '.jpg', '.png'))]
        self.labels = [1] * len(self.image_paths)  # Change label to 1 for unhealthy images
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image_path, label):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            image = cv2.resize(image, self.resize_shape)  # resize the image
        except:
            print("Failed to resize image:", image_path)
            image = np.zeros(self.resize_shape)  # return black image if resize fails
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)
        image = image / 255.0
        return image, label

    def generator(self):
        for image_path, label in zip(self.image_paths, self.labels):
            yield self.preprocess_image(image_path, label)


class NormalDataset(tf.keras.utils.Sequence):
    def __init__(self, image_folder, resize_shape=(256, 256)):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(self.image_folder, file) for file in os.listdir(self.image_folder) if
                            file.endswith(('.jpeg', '.jpg', '.png'))]
        self.labels = [0] * len(self.image_paths)
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image_path, label):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            image = cv2.resize(image, self.resize_shape)  # resize the image
        except:
            print("Failed to resize image:", image_path)
            image = np.zeros(self.resize_shape)  # return black image if resize fails
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=-1)
        image = image / 255.0
        return image, label

    def generator(self):
        for image_path, label in zip(self.image_paths, self.labels):
            yield self.preprocess_image(image_path, label)
