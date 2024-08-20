# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 23:09:01 2023

@author: hasso
"""

from flask import Flask, render_template, request
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Anomaly_detaction import AnomalyDetector
from VQVAE import VQVAE
import tensorflow as tf
app = Flask(__name__)



# Define hyperparameters
batch_size = 32
num_epochs = 50
initial_learning_rate = 0.001
num_embeddings = 512
embedding_dim = 128
commitment_cost = 0.25

dummy_input = tf.zeros((1, 256, 256, 1))
vqvae = VQVAE(in_channels=1, num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost)
vqvae(dummy_input)
vqvae.load_model('model_f.h5')
vqvae_model = vqvae
threshold = 0.04

detector = AnomalyDetector(vqvae_model, threshold)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file.stream).convert("L")  # Convert image to grayscale
        img_array = np.array(img)[..., np.newaxis]  # Add a channel dimension

        x = np.expand_dims(img_array, axis=0)  # Add a batch dimension
        x_recon, _, _, _, _ = detector.vqvae_model(x)
        detector.visualize_heatmap(x[0], x_recon[0])

        # Save the anomaly overlay with a circle as an image
        fig = detector.visualize_heatmap(x[0], x_recon[0])
        buf = BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        image_data = base64.b64encode(buf.read()).decode("utf-8")

        return render_template("result.html", image_data=image_data)
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
