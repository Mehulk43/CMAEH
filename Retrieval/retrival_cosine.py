from tools import *
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import models_mae  # Import the models_mae module if prepare_model function is defined there

# # Define transformations for the query image
# transform_query = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# Define transformations for the test dataset
transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_config():
    config = {
             "dataset": "cifar10",
            #"dataset": "cifar10-2",
            #"dataset": "coco",
            #"dataset": "nuswide_21",
            # "dataset": "imagenet",
            "batch_size" : 32, "crop_size":224
    }
    config = config_dataset(config)
    return config
config = get_config()
train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset,test_dataset,databse_dataset = get_data(config)

# Load the trained model
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model state dictionary
    checkpoint = torch.load(chkpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(checkpoint, strict=False)
    return model

model_path = 'trained_mae_model_0.50.pth'
model = prepare_model(model_path)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)  # Move model to GPU if available

# testset = test_dataset

# print(test_dataset)

# # Load the test dataset
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# print("testset is ",testset)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Select a query image index from the test dataset
# Select a query image index from the test dataset
query_index = 999 # Change this index to select a different query image
query_image = test_dataset[query_index][0]
query_image = query_image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU
print(query_image.shape)
# query_image = query_image.permute(1, 2, 0)  # Permute dimensions for plotting
# query_image = query_image.clamp(0, 1)
# plt.imshow(query_image.cpu())  # Move to CPU and plot
# query_image = query_image.unsqueeze(0).to(device)  # Add batch dimension and move to GPU
# # plt.show()
# # plt.imshow(query_image, cmap='gray').cpu()
# print(query_image.shape)
# exit()
# Pass the query image through the model to obtain its feature representation
with torch.no_grad():
    query_features, _, _ = model.forward_encoder(query_image, mask_ratio=0.5)

# Calculate cosine similarity between the query image's features and all test images' features
similarities = []
for images, _,_ in test_loader:
    images = images.to(device)
    with torch.no_grad():
        features, _, _ = model.forward_encoder(images, mask_ratio=0.5)
    similarities.extend(torch.cosine_similarity(query_features, features, dim=1).cpu().numpy())

# Retrieve top-k most similar images
# Retrieve top-k most similar images
k = 10  # Number of similar images to retrieve
top_indices = np.argsort(similarities)[-k:][::-1].ravel()  # Ensure top_indices is 1D
print(top_indices)
# Visualize the query image and retrieved images
plt.figure(figsize=(12, 6))
plt.subplot(1, k+1, 1)
query_image_np = query_image.squeeze().cpu().permute(1, 2, 0).numpy()
plt.imshow(np.clip(query_image_np, 0, 1))  # Clamp pixel values
plt.title('Query Image')

for i, idx in enumerate(top_indices):
    idx = int(idx)
    # similar_image, target,_ = test_dataset[idx]  # Retrieve image and label separately
    if idx < len(databse_dataset):
        similar_image, target,_ = databse_dataset[idx]  # Retrieve image and label separately
    else:
        print(f"Index {idx} is out of bounds for the dataset.")
    
    
    if i < k:  # Ensure subplot index stays within range
        plt.subplot(1, k+1, i+2)  # Adjust subplot index
        similar_image_np = similar_image.permute(1, 2, 0).numpy()
        plt.imshow(np.clip(similar_image_np, 0, 1))  # Clamp pixel values
        # plt.title(f'Target: {target}')

plt.show()

# for i, idx in enumerate(top_indices):
#     idx = int(idx)
#     # similar_image, target,_ = test_loader[idx]  # Retrieve image and label separately
#     for images, _, _ in test_loader:
#         # Get the image corresponding to the index
#         # similar_image = images[idx]
#         similar_image = None
#         if 0 <= idx < len(images):
#             similar_image = images[idx]
#             similar_image_np = similar_image.permute(1, 2, 0).numpy()
#         else:
#             print(f"Index {idx} is out of bounds for dimension 0 with size {len(images)}")
#         # Plot the image
#         if i < k:  # Ensure subplot index stays within range
#             plt.subplot(1, k+1, i+2)  # Adjust subplot index
#             similar_image_np = similar_image.permute(1, 2, 0).numpy()
#             plt.imshow(np.clip(similar_image_np, 0, 1))  # Clamp pixel values
#             plt.title(f'Top-{i+1}')
#             # plt.title(f'Target: {target}')

# plt.show()