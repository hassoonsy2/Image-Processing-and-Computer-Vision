import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#ToDo : Implement Insaption Block !
#ToDo : Make documentaion

class VQVAE(keras.Model):
    def __init__(self, d, n_channels, code_size, n_block=3, n_res_block=4, cond_channels=None, dropout_p=.5,
                 reconstruction_loss=keras.losses.MeanAbsoluteError()):
        super().__init__()
        self.code_size = code_size
        self.d = d
        self.cond_channels = cond_channels
        self.reconstruction_loss = reconstruction_loss

        if isinstance(n_channels, int):
            n_channels = [n_channels] * (n_block + 1)
        else:
            n_block = len(n_channels) - 1

        # Encoder
        down = [layers.Conv2D(n_channels[0], kernel_size=7, padding='same'),
                layers.BatchNormalization()]

        for block in range(n_block):
            for res_block in range(n_res_block):
                down.append(nn_blocks.GatedResNet(n_channels[block], 3, dropout_p=dropout_p, conv=layers.Conv2D,
                                                  norm=layers.BatchNormalization))

            down.extend([layers.Conv2D(n_channels[block + 1], kernel_size=5, strides=2, padding='same'),
                         layers.BatchNormalization()])

        down.append(nn_blocks.GatedResNet(n_channels[-1], 3, dropout_p=dropout_p, conv=layers.Conv2D,
                                          norm=layers.BatchNormalization()))

        self.Q = keras.Sequential(down)

        self.codebook = nn_blocks.Quantize(code_size, n_channels[-1])

        # Decoder
        up = [layers.Conv2D(n_channels[-1], kernel_size=3, padding='same'),
              layers.BatchNormalization()]
        for block in range(n_block):
            for res_block in range(n_res_block):
                up.append(nn_blocks.GatedResNet(n_channels[-(block + 1)], 3, dropout_p=dropout_p,
                                                conv=layers.Conv2D, norm=layers.BatchNormalization))

            up.extend([layers.Conv2DTranspose(n_channels[-(block + 2)], kernel_size=6, strides=2, padding='same'),
                       layers.BatchNormalization()])

        up.append(nn_blocks.GatedResNet(n_channels[0], 3, dropout_p=dropout_p, conv=layers.Conv2D,
                                        norm=layers.BatchNormalization()))

        up.extend([layers.ELU(),
                   layers.Conv2DTranspose(d, kernel_size=1, padding='same')])
        self.P = keras.Sequential(up)

    def encode(self, x, cond=None):
        out = tf.expand_dims(x, axis=1)

        if self.cond_channels is not None:
            cond = tf.cast(cond, dtype=tf.float32)
            if len(cond.shape) == 2:
                cond = tf.reshape(cond, [cond.shape[0], -1, 1, 1])
                cond = tf.broadcast_to(cond, [-1, -1, x.shape[1], x.shape[2]])
            out = tf.concat([out, cond], 1)

        return self.Q(out)

    def decode(self, latents, cond=None):
        if self.cond_channels is not None:
            cond = tf.cast(cond, dtype=tf.float32)
            if len(cond.shape) == 2:
                cond = tf.reshape(cond, [cond.shape[0], -1, 1, 1])
                cond = tf.broadcast_to(cond, [-1, -1, latents.shape[2], latents.shape[3]])
            tf.concat([latents, cond], axis=-1)
            return latents

    def forward(self, x, cond=None):
        z = self.encode(x, cond)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decode(e_st, cond)

        diff1 = tf.reduce_mean(tf.square(z - tf.stop_gradient(e)))
        diff2 = tf.reduce_mean(tf.square(e - tf.stop_gradient(z)))
        return x_tilde, diff1 + diff2

    def loss(self, x, cond=None, reduction='mean'):
        x_tilde, diff = self.forward(x, cond)
        x = tf.expand_dims(x, axis=1)
        recon_loss = self.reconstruction_loss(x_tilde, x, reduction=reduction)

        if reduction == 'mean':
            loss = recon_loss + diff

        elif reduction == 'none':
            loss = tf.reduce_mean(recon_loss) + diff

        return {'loss': loss, 'recon_loss': recon_loss, 'reg_loss': diff}
