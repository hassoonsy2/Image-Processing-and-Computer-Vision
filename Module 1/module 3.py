import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import AffineTransform, warp

def plot_before_and_after(before, after):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(before)
    axes[0].set_title('Before')
    axes[1].imshow(after)
    axes[1].set_title('After')
    plt.show()

# Laad een voorbeeldafbeelding
image = data.checkerboard()

# Rotatie
rotation_transform = AffineTransform(rotation=np.pi / 4)  # 45 graden rotatie
rotated_image = warp(image, rotation_transform)
plot_before_and_after(image, rotated_image)

# Translatie
translation_transform = AffineTransform(translation=(50, 50))  # Verplaats 50 pixels naar rechts en 50 naar beneden
translated_image = warp(image, translation_transform)
plot_before_and_after(image, translated_image)

# Schaalvergroting (stretch)
scaling_transform = AffineTransform(scale=(1.5, 0.5))  # Horizontaal uitrekken met een factor 1.5 en verticaal comprimeren met een factor 0.5
scaled_image = warp(image, scaling_transform)
plot_before_and_after(image, scaled_image)
