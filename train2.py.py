from tools import *
import torch
import time
import os
import json
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import models_mae
torch.multiprocessing.set_sharing_strategy('file_system')
from util.pos_embed import interpolate_pos_embed

def get_config():
    config = {
            "dataset": "cifar10",
            #"dataset": "cifar10-2",
            #"dataset": "coco",
            #"dataset": "nuswide_21",
            # "dataset": "imagenet",
            "batch_size" : 32, "crop_size":224,
            "device": torch.device("cuda:1"),"alpha": 0.1,"step_continuation": 20,"info": "Masked_mae",
            "resize_size": 256,"epoch": 150, "test_map": 3,"save_path": "save/HashNet"
    }
    config = config_dataset(config)
    return config
# Define the function to prepare the model
def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model state dictionary
    checkpoint = torch.load(chkpt_dir, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    if hasattr(model, 'hash_layer'):
    # Initialize weights for hash_layer, assuming it's a Linear layer for this example
        nn.init.xavier_uniform_(model.hash_layer[1].weight)
        nn.init.constant_(model.hash_layer[1].bias, 0)
        nn.init.xavier_uniform_(model.hash_layer[3].weight)
        nn.init.constant_(model.hash_layer[3].bias, 0)
    else:
        print("hash_layer not found in model")

    # Check and initialize the head layer if it has been modified or is new
    if hasattr(model, 'head'):
        nn.init.xavier_uniform_(model.head.weight)
        nn.init.constant_(model.head.bias, 0)
    else:
        print("head layer not found in model")
    return model

# f = open(results_path, 'a')
def train_val(config):
    # config = get_config()
    print(config)
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset,test_dataset,databse_dataset = get_data(config)
    config["num_train"] = num_train
    config["n_class"] = 10
# Define your training parameters
    device = config["device"]
    print(device)
    
    # num_epochs = 100
    learning_rate = 1e-5
    train_loss=0
    train_mixed_loss = 0




    # Load the model
    chkpt_dir = 'mae_visualize_vit_large.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    model_mae=model_mae.to(device)  # Move model to GPU if available
    # print(model_mae)
    # exit()
    bit = 64 # change the number code length
    
    # Define loss function and optimizer
    criterion = HashNetLoss(config, bit)
    optimizer = optim.Adam(model_mae.parameters(), lr=learning_rate)

    # Set model to train mode
    model_mae.train()
    
    # for param in model_mae.parameters():
    #         param.requires_grad = False


    # Training loop
    Best_mAP = 0
    for epoch in range(config["epoch"]):
        criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5
        
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"], criterion.scale), end="")
        
        running_loss = 0.0
        for images, labels, Index in tqdm(train_loader):
            images = images.to(device)  # Move data to GPU if available
            labels = labels.to(device) # Move data to GPU if available
            optimizer.zero_grad()
        
            loss, pred, mask, hash_code = model_mae.forward(images,mask_ratio = 0.25)
            # print(loss,hash_code.shape)
            loss1 = criterion(hash_code, labels.float(), Index,config)
            train_loss += loss1.item()
        # Backward pass and optimization
            mixed_loss =  0.3*loss + 0.7*loss1
            train_mixed_loss += mixed_loss.item() 
            
            # print("loss.requires_grad:", loss.requires_grad)
            # print("loss1.requires_grad:", loss1.requires_grad)

            
            mixed_loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_mixed_loss = train_mixed_loss / len(train_loader)
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['epoch']}], Loss: {epoch_loss:.4f}, Loss: {train_loss:.4f}")

        # print(f"Epoch [{epoch+1}/{config["epoch"]}], Loss: {epoch_loss:.4f}, Loss: {train_loss:.4f}")

        
        print("\b\b\b\b\b\b\b Mixed loss:%.5f train_loss: %.5f recon+loss: %.5f" %
              (train_mixed_loss, train_loss, loss))
        # f.write('Train | Epoch: %d | Mixed loss:%.5f | loss:%.5f | contloss:%.5f\n' % (
        #     epoch, train_mixed_loss, train_loss, loss))
        
        if (epoch + 1) % config["test_map"] == 0:
            Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, model_mae, bit, epoch, num_dataset)
            
# Save the trained model
    torch.save(model_mae.state_dict(), 'trained_mae_model.pth')
    print("Training completed and model saved.")
    # f.close()
class HashNetLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

        self.scale = 1

    def forward(self, u, y, ind, config):
        
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = config["alpha"] * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        # weight
        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss

if __name__ == "__main__":
    config = get_config()
    config["pr_curve_path"] = f"log/mae/MASKED_{config['dataset']}_{64}.json"
    train_val(config)
