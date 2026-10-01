import numpy as np
import matplotlib.pyplot as plt
import struct
import pandas as pd
from sklearn.decomposition import PCA

def load_mnist_images_raw(filepath):
    with open(filepath, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows * cols)
        return images

def load_mnist_labels_raw(filepath):
    with open(filepath, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

X_train = load_mnist_images_raw('mnist_data/train-images.idx3-ubyte')
y_train = load_mnist_labels_raw('mnist_data/train-labels.idx1-ubyte')

# Normalized pixels
X_scaled = X_train.astype(np.float64) / 255.0

# number of principal components
n_components = 30
pca = PCA(n_components=n_components)
pca.fit(X_scaled)


eigenvalues = pca.explained_variance_     
eigenvectors = pca.components_            
mean_image = pca.mean_                    

# Top 30 Eigenvalues
plt.figure(figsize=(8, 4))
plt.plot(range(1, 31), eigenvalues, 'o-', color='blue', linewidth=2)
plt.title('Top 30 Eigenvalues of Global MNIST PCA', fontsize=12)
plt.xlabel('Principal Component Index', fontsize=10)
plt.ylabel('Eigenvalue (Variance)', fontsize=10)
plt.xticks(range(1, 31, 2))
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('plots/top_30_eigenvalues.png')
# plt.show()

# Display all 30 Eigenvectors as Images (5 rows x 6 columns)
fig, axes = plt.subplots(5, 6, figsize=(12, 10))
for i, ax in enumerate(axes.flat):
    # Reshape each 784 eigenvector into 28x28 grid
    ax.imshow(eigenvectors[i].reshape(28, 28), cmap='seismic')
    ax.set_title(f'PC {i+1}', fontsize=9)
    ax.axis('off')

plt.suptitle('Top 30 Eigenvectors (Eigen-Digits)', fontsize=14, y=0.98)
plt.tight_layout()
plt.savefig('plots/top_30_eigenvectors.png')
# plt.show()

# Find the first index for each digit class (0 through 9)
digit_indices = [np.where(y_train == d)[0][0] for d in range(10)]

k_values = [2, 5, 20, 30]
mse_table = {f"k={k}": [] for k in k_values}
reconstructions = {d: {} for d in range(10)}

for digit in range(10):
    idx = digit_indices[digit]
    x_orig = X_scaled[idx]
    x_centered = x_orig - mean_image
    
    for k in k_values:
        # top k eigenvectors
        W_k = eigenvectors[:k]  

        # dot product to determine similarity
        z = np.dot(x_centered, W_k.T) 
        
        # Reconstruct original image
        x_reconstructed = mean_image + np.dot(z, W_k)
        
        mse = np.mean((x_orig - x_reconstructed) ** 2)
        mse_table[f"k={k}"].append(mse)
        
        reconstructions[digit][k] = x_reconstructed

# MSE Results Table
df_mse = pd.DataFrame(mse_table, index=[f"Digit {d}" for d in range(10)])
with open('plots/mse_results.csv', 'w') as f:
    df_mse.to_csv(f)

# Visual Comparison: Original vs Reconstructed Images for Digits 0-9
fig, axes = plt.subplots(10, 5, figsize=(10, 18))
column_titles = ["Original", "k=2", "k=5", "k=20", "k=30"]

for col_idx, title in enumerate(column_titles):
    axes[0, col_idx].set_title(title, fontsize=12, fontweight='bold')

for digit in range(10):
    idx = digit_indices[digit]
    
    axes[digit, 0].imshow(X_scaled[idx].reshape(28, 28), cmap='gray')
    axes[digit, 0].set_ylabel(f"Digit {digit}", fontsize=11, fontweight='bold')
    axes[digit, 0].set_xticks([])
    axes[digit, 0].set_yticks([])
    
    for col_idx, k in enumerate(k_values, start=1):
        reconstructed_img = reconstructions[digit][k].reshape(28, 28)
        axes[digit, col_idx].imshow(reconstructed_img, cmap='gray')
        axes[digit, col_idx].axis('off')

plt.tight_layout()
plt.savefig('plots/reconstructed_images_comparison.png')
plt.show()