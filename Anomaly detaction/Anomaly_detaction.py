import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class AnomalyDetector:
    def __init__(self, vqvae_model, threshold):
        self.vqvae_model = vqvae_model
        self.threshold = threshold

    def detect_anomalies(self, dataloader):
        anomalies = []
        for batch in dataloader:
            x = batch[0].numpy()
            x_recon, _, _, _, _ = self.vqvae_model(x)
            x_recon = x_recon.numpy()
            reconstruction_error = tf.reduce_mean((x - x_recon) ** 2, axis=[1, 2, 3])
            anomaly_indices = tf.squeeze(tf.where(reconstruction_error > self.threshold))
            if anomaly_indices.ndim == 0:
                anomalies.append(anomaly_indices.numpy().tolist())
            else:
                anomalies.extend(anomaly_indices.numpy().tolist())
        return anomalies

    def visualize_heatmap(self, x, x_recon, figsize=(10, 4)):
        x = x.numpy().transpose(1, 2, 0)
        x_recon = x_recon.numpy().transpose(1, 2, 0)

        heatmap = np.mean(np.square(x - x_recon), axis=-1)
        mask = heatmap > self.threshold

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(x.squeeze(), cmap='gray')
        axes[0].set_title('Original Image')
        axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[1].set_title('Heatmap')
        axes[2].imshow(x.squeeze(), cmap='gray')
        axes[2].imshow(mask, cmap='jet', alpha=0.5, interpolation='nearest')
        axes[2].set_title('Anomaly Overlay')

        for ax in axes:
            ax.axis('off')

        plt.show()

