import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import models_mae 
import sklearn.neighbors
from sklearn.neighbors import NearestNeighbors

# Define transformations for the query image
transform_query = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the trained model
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model state dictionary
    checkpoint = torch.load(chkpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint, strict=False)
    return model

model_path = 'trained_mae_model.pth'
model = prepare_model(model_path)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)  # Move model to GPU if available

# Define transformations for the test dataset
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Extract features for all test images
test_features = []
test_labels = []
for images, labels in testloader:
    images = images.to(device)
    with torch.no_grad():
        features, _, _ = model.forward_encoder(images, mask_ratio=0.75)
    test_features.append(features.cpu().numpy())
    test_labels.extend(labels.numpy())
test_features = np.concatenate(test_features, axis=0)

# Flatten the test features array to 2D
test_features_flattened = test_features.reshape(test_features.shape[0], -1)

# Fit Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(test_features_flattened)

# Fit Nearest Neighbors model
# knn = NearestNeighbors(n_neighbors=5, metric='cosine')
# knn.fit(test_features)

# Select a query image index from the test dataset
query_index = 800  # Change this index to select a different query image
query_image, _ = testset[query_index]
query_image = query_image.unsqueeze(0).to(device)

# Extract features for the query image
with torch.no_grad():
    query_features, _, _ = model.forward_encoder(query_image, mask_ratio=0.75)
query_features = query_features.cpu().numpy()

# Flatten the query features array to 2D
query_features_flattened = query_features.reshape(1, -1)

# Find k nearest neighbors
distances, indices = knn.kneighbors(query_features_flattened)

# Find k nearest neighbors
# distances, indices = knn.kneighbors(query_features)

# Visualize the query image and retrieved images
plt.figure(figsize=(12, 6))
plt.subplot(1, len(indices[0])+1, 1)
query_image_np = query_image.squeeze().cpu().permute(1, 2, 0).numpy()
plt.imshow(np.clip(query_image_np, 0, 1))  # Clamp pixel values
plt.title('Query Image')

for i, idx in enumerate(indices[0]):
    similar_image, target = testset[idx]  # Retrieve image and label separately
    plt.subplot(1, len(indices[0])+1, i+2)
    similar_image_np = similar_image.permute(1, 2, 0).numpy()
    plt.imshow(np.clip(similar_image_np, 0, 1))  # Clamp pixel values
    plt.title(f'Target: {target}')

plt.show()
