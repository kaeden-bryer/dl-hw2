import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_mnist_images_raw(filepath):
    with open(filepath, 'rb') as f:
        # Read 16-byte header
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        # Read pixel data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images

def load_mnist_labels_raw(filepath):
    with open(filepath, 'rb') as f:
        # Read 8-byte header
        magic, num_labels = struct.unpack('>II', f.read(8))
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

X_train = load_mnist_images_raw('mnist_data/train-images.idx3-ubyte')
y_train = load_mnist_labels_raw('mnist_data/train-labels.idx1-ubyte')

# # Pick the first image (index 0)
# sample_index = 0
# sample_vector = X_train[sample_index]           # Shape: (784,)
# sample_matrix = sample_vector.reshape(28, 28)  # Shape: (28, 28)

# df_image = pd.DataFrame(sample_matrix)

# with open('sample_image.csv', 'w') as f:
#     df_image.to_csv(f, index=False)

# Plot n example images with their labels
num_images = 20
fig, axes = plt.subplots(1, num_images, figsize=(12, 3))

for i in range(num_images):
    # Select image i and reshape from 784 1D vector to 28x28 2D grid
    img = X_train[i].reshape(28, 28)
    label = y_train[i]
    
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"{label}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()