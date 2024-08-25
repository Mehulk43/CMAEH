import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import models_mae
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Define the transformations for the CIFAR-10 test dataset
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the function to prepare the model
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    return model

# Load the trained model
chkpt_dir = 'trained_mae_model.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_mae = model_mae.to(device)  # Move model to the same device as input data

# Download CIFAR-10 test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=2)

# Function to encode test images and measure runtime
def encode_images():
    start_time = time.time()
    encoded_images = []
    with torch.no_grad():
        for images, _ in tqdm(testloader, desc='Encoding Test Images'):
            images = images.to(device)
            x, _, _ = model_mae.forward_encoder(images, mask_ratio=0.25)
            encoded_images.append(x.cpu().numpy())
    encoded_images = np.concatenate(encoded_images, axis=0)
    end_time = time.time()
    print(f"Encoding runtime: {end_time - start_time:.2f} seconds")
    return encoded_images

encoded_images = encode_images()

# Flatten the encoded images
encoded_images_flat = encoded_images.reshape(encoded_images.shape[0], -1)

# Function to fit the k-NN model and measure runtime
def fit_knn_model(n_neighbors):
    start_time = time.time()
    knn_model = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')  # 1 query + n_neighbors similar
    knn_model.fit(encoded_images_flat)
    end_time = time.time()
    print(f"k-NN fitting runtime: {end_time - start_time:.2f} seconds")
    return knn_model

# Function to retrieve and visualize similar images
def retrieve_and_visualize_similar_images(knn_model, image_index, n_neighbors):
    start_time = time.time()
    query_image = encoded_images_flat[image_index].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_image)
    end_time = time.time()
    print(f"Retrieval runtime: {end_time - start_time:.2f} seconds")

    num_cols = 11  # 1 query image + 10 similar images per row
    num_rows = (n_neighbors + 10) // 10

    plt.figure(figsize=(num_cols * 2, num_rows * 2.5))
    
    # Display query image on the left side
    plt.subplot(num_rows, num_cols, 1)
    query_image = testset[image_index][0].permute(1, 2, 0).numpy()  # Convert to numpy array and rearrange dimensions
    query_image = (query_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Unnormalize
    query_image = np.clip(query_image, 0, 1)  # Clip to valid range
    plt.imshow(query_image)
    plt.title('Query Image')
    plt.axis('off')

    # Display retrieved images on the right side
    for i, idx in enumerate(indices[0][1:n_neighbors + 1]):  # Start from 1 to skip the query image
        row = i // 10
        col = (i % 10) + 1  # Shift columns to the right
        plt.subplot(num_rows, num_cols, (row * num_cols) + col + 1)
        similar_image = testset[idx][0].permute(1, 2, 0).numpy()
        similar_image = (similar_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Unnormalize
        similar_image = np.clip(similar_image, 0, 1)  # Clip to valid range
        plt.imshow(similar_image)
        plt.title(f'Similar {i+1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
# def retrieve_and_visualize_similar_images(knn_model, image_index, n_neighbors):
#     start_time = time.time()
#     query_image = encoded_images_flat[image_index].reshape(1, -1)
#     distances, indices = knn_model.kneighbors(query_image)
#     end_time = time.time()
#     print(f"Retrieval runtime: {end_time - start_time:.2f} seconds")

#     plt.figure(figsize=(20, 5))
    
#     # Display query image on the left side
#     plt.subplot(1, n_neighbors + 1, 1)
#     query_image = testset[image_index][0].permute(1, 2, 0).numpy()  # Convert to numpy array and rearrange dimensions
#     query_image = (query_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Unnormalize
#     query_image = np.clip(query_image, 0, 1)  # Clip to valid range
#     plt.imshow(query_image)
#     plt.title('Query Image')
#     plt.axis('off')

#     # Display retrieved images on the right side
#     for i, idx in enumerate(indices[0][1:]):  # Start from 1 to skip the query image
#         similar_image = testset[idx][0].permute(1, 2, 0).numpy()
#         similar_image = (similar_image * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Unnormalize
#         similar_image = np.clip(similar_image, 0, 1)  # Clip to valid range
#         plt.subplot(1, n_neighbors + 1, i + 2)
#         plt.imshow(similar_image)
#         plt.title(f'Similar {i+1}')
#         plt.axis('off')

#     plt.tight_layout()
#     plt.show()

# Retrieve and visualize similar images for a query image
query_image_index = 500  # Change this index to visualize similar images for a different query image
n_neighbors_options = [5, 10, 15, 20, 25]

for n_neighbors in n_neighbors_options:
    print(f"\nVisualizing {n_neighbors} similar images:")
    knn_model = fit_knn_model(n_neighbors)
    retrieve_and_visualize_similar_images(knn_model, query_image_index, n_neighbors)
