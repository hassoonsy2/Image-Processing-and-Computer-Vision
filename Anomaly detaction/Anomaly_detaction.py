import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor
from matplotlib.patches import Circle
import skimage.measure
import skimage.draw

class AnomalyDetector:
    def __init__(self, vqvae_model, threshold):
        self.vqvae_model = vqvae_model
        self.threshold = threshold


    def detect_anomalies_for_batch(self, batch):
        x = batch[0].numpy()
        x_recon, _, _, _, _ = self.vqvae_model(x)
        x_recon = x_recon.numpy()
        reconstruction_error = tf.reduce_mean((x - x_recon) ** 2, axis=[1, 2, 3])
        anomaly_indices = tf.squeeze(tf.where(reconstruction_error > self.threshold))
        if anomaly_indices.ndim == 0:
            return [anomaly_indices.numpy().tolist()]
        else:
            return anomaly_indices.numpy().tolist()

    def detect_anomalies_on_samples(self, samples):
        anomalies = []
        for x, _ in samples:
            print(anomalies)
            x = x[np.newaxis, ...]
            x_recon, _, _, _, _ = self.vqvae_model(x)
            x_recon = x_recon.numpy()
            reconstruction_error = tf.reduce_mean((x - x_recon) ** 2, axis=[1, 2, 3])
            anomaly_indices = tf.squeeze(tf.where(reconstruction_error > self.threshold))
            if anomaly_indices.ndim == 0:
                anomalies.append(anomaly_indices.numpy().tolist())
            else:
                anomalies.extend(anomaly_indices.numpy().tolist())
        return anomalies

    def detect_anomalies(self, dataloader, num_workers=4):
        anomalies = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_batch = {executor.submit(self.detect_anomalies_for_batch, batch): batch for batch in dataloader}
            for future in concurrent.futures.as_completed(future_to_batch):
                try:
                    batch_anomalies = future.result()
                    anomalies.extend(batch_anomalies)
                except Exception as exc:
                    print(f"An error occurred while processing a batch: {exc}")

        return anomalies

    def normalize_image(self,img):
        # Check if the image has a valid range
        if np.max(img) == np.min(img):
            return img
        # Normalize the image to the range [0, 1]
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def visualize_heatmap(self, x, x_recon, figsize=(10, 4), resize_shape=None):
        if isinstance(x, tf.Tensor):
            x = x.numpy()
        if isinstance(x_recon, tf.Tensor):
            x_recon = x_recon.numpy()

        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)

        if x_recon.ndim == 2:
            x_recon = np.expand_dims(x_recon, axis=-1)  # Ensure x_recon also has a third dimension

        if resize_shape is not None:
            x = tf.image.resize(x, resize_shape)
            x_recon = tf.image.resize(x_recon, resize_shape)

        x = self.normalize_image(x)
        x_recon = self.normalize_image(x_recon)

        heatmap = np.mean(np.square(x - x_recon), axis=-1)
        mask = heatmap > self.threshold

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(x.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')

        for i, j in zip(*np.where(mask)):
            circle = Circle((j, i), radius=10, color='red', fill=False)
            axes[0].add_patch(circle)

        axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[1].set_title('Heatmap')
        axes[2].imshow(x.squeeze(), cmap='gray')
        axes[2].imshow(mask, cmap='jet', alpha=0.5, interpolation='nearest')
        axes[2].set_title('Anomaly Overlay')

        for ax in axes:
            ax.axis('off')

        plt.show()





