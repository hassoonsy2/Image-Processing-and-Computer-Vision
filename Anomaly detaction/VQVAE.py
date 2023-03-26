import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model

class ResidualBlock(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(out_channels, kernel_size=3, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv2 = layers.Conv2D(out_channels, kernel_size=3, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()

    def call(self, inputs):
        residual = inputs
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu2(x)
        return x

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        initial_embeddings = tf.random.uniform((num_embeddings, embedding_dim), minval=-1/num_embeddings, maxval=1/num_embeddings)
        self.embeddings = tf.Variable(initial_embeddings, trainable=True)

    def call(self, z_e):
        z_e = tf.transpose(z_e, perm=[0, 3, 1, 2])
        z_e_flat = tf.reshape(z_e, [-1, self.embedding_dim])
        distances = tf.norm(tf.expand_dims(z_e_flat, axis=1) - self.embeddings, axis=-1)

        indices = tf.argmin(distances, axis=1)
        z_q = tf.gather(self.embeddings, indices)
        z_q = tf.reshape(z_q, tf.shape(z_e))
        z_q_with_gradient = z_q + tf.stop_gradient(z_e - z_q)

        e_latent_loss = tf.reduce_mean((tf.stop_gradient(z_q) - z_e) ** 2)
        q_latent_loss = tf.reduce_mean((z_q_with_gradient - tf.stop_gradient(z_e)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        return tf.transpose(z_q_with_gradient, perm=[0, 2, 3, 1]), indices, loss

class VQVAE(Model):
    def __init__(self, in_channels, num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=4, strides=2, padding='same', activation='relu', input_shape=(None, None, in_channels)),
            layers.Conv2D(64, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.Conv2D(128, kernel_size=4, strides=2, padding='same', activation='relu'),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        ])
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = tf.keras.Sequential([
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', activation='relu'),
            layers.Conv2DTranspose(in_channels, kernel_size=4, strides=2, padding='same', activation='sigmoid'),
        ])

    def call(self, inputs):
        z_e = self.encoder(inputs)
        z_q, indices, loss = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, z_e, z_q, indices, loss

    def train_vqvae(self, dataloader, num_epochs, initial_learning_rate, dataset_size, batch_size, print_every=1):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=dataset_size // batch_size,
            decay_rate=0.9,
            staircase=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        self.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)
        training_losses = []

        steps_per_epoch = dataset_size // batch_size

        # Set up the matplotlib interactive mode
        plt.ion()
        fig, ax = plt.subplots()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for i, (x, _) in enumerate(dataloader):
                with tf.GradientTape() as tape:
                    x_recon, z_e, z_q, indices, vq_loss = self(x)
                    recon_loss = tf.reduce_mean(tf.keras.losses.MSE(x_recon, x))
                    loss = recon_loss + vq_loss

                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                epoch_loss += loss.numpy()

            avg_epoch_loss = epoch_loss / steps_per_epoch
            training_losses.append(avg_epoch_loss)

            if (epoch + 1) % print_every == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")

            # Update the learning curve after every epoch
            ax.clear()
            ax.plot(range(1, epoch + 2), training_losses)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title("Training Loss Curve")
            plt.draw()
            plt.pause(0.1)

        # Turn off the interactive mode
        plt.ioff()

        return training_losses

    def save_model(self, path):
        self.save_weights(path)

    def load_model(self, path):
        self.load_weights(path)

