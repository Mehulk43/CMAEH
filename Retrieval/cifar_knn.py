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

# Encode test images
encoded_images = []
with torch.no_grad():
    for images, _ in tqdm(testloader, desc='Encoding Test Images'):
        images = images.to(device)
        x, _, _ = model_mae.forward_encoder(images, mask_ratio=0.25)
        encoded_images.append(x.cpu().numpy())

encoded_images = np.concatenate(encoded_images, axis=0)

# Flatten the encoded images
encoded_images_flat = encoded_images.reshape(encoded_images.shape[0], -1)

# Fit a Nearest Neighbors model
knn_model = NearestNeighbors(n_neighbors=6, algorithm='auto')
knn_model.fit(encoded_images_flat)

# Function to retrieve and visualize similar images
def retrieve_and_visualize_similar_images(image_index):
    query_image = encoded_images_flat[image_index].reshape(1, -1)
    distances, indices = knn_model.kneighbors(query_image)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 6, 1)
    query_image = testset[image_index][0].permute(1, 2, 0)  # Convert to numpy array and rearrange dimensions
    plt.imshow(query_image)
    plt.title('Query Image')
    plt.axis('off')

    for i, idx in enumerate(indices[0][1:]):  # Start from 1 to skip the query image
        similar_image = testset[idx][0].permute(1, 2, 0)
        plt.subplot(1, 6, i + 2)
        plt.imshow(similar_image)
        plt.title(f'Similar {i+1}')
        plt.axis('off')

    plt.show()

# Retrieve and visualize similar images for a query image
query_image_index = 500
# Change this index to visualize similar images for a different query image
retrieve_and_visualize_similar_images(query_image_index)
