from tools import *
import torch
import time
import os
import json
import torchvision
import seaborn as sns
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import models_mae
torch.multiprocessing.set_sharing_strategy('file_system')
from util.pos_embed import interpolate_pos_embed
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def get_config():
    config = {
            "dataset": "cifar10",
            #"dataset": "cifar10-2",
            #"dataset": "coco",
            #"dataset": "nuswide_21",
            # "dataset": "imagenet",
            "batch_size" : 32, "crop_size":224,
            "device": torch.device("cuda:1"),"alpha": 0.08,"step_continuation": 20,"info": "Masked_mae",
            "resize_size": 256,"epoch": 150, "test_map": 30,"save_path": "save/HashNet"
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
    # print(msg)
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

def train_val(config):
    # config = get_config()
    print(config)
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset,test_dataset,databse_dataset = get_data(config)
    config["num_train"] = num_train
    # config["n_class"] = 10
# Define your training parameters
    device = config["device"]
    print(device)
    
    # num_epochs = 100
    learning_rate = 1e-5
    
    
    # Load the model
    chkpt_dir = 'mae_visualize_vit_large.pth' # Load the pretrained weight directory
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    model_mae=model_mae.to(device)  # Move model to GPU if available
    # print(model_mae)
    # exit()
    bit = 64 # change the number code length
    
    # Define loss function and optimizer
    criterion = HashNetLoss(config, bit)
    criterion_contrastive = ContrastiveLoss()
    optimizer = optim.Adam(model_mae.parameters(), lr=learning_rate)

    
    start_epoch = 1
    for epoch in range(start_epoch,config["epoch"]+1):
        # criterion.scale = (epoch // config["step_continuation"] + 1) ** 0.5
        
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        
        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch, config["epoch"], current_time, bit, config["dataset"]), end="")
        # Set model to train mode
        model_mae.train()
        # Training loop
        train_loss=0
        train_mixed_loss = 0
        train_loss_contrastive = 0
        Best_mAP = 0
        running_loss = 0.0
        for images, labels, Index in train_loader:
            images = images.to(device)  # Move data to GPU if available
            labels = labels.to(device) # Move data to GPU if available
            optimizer.zero_grad()
        
            loss, pred, mask, hash_code = model_mae(images,mask_ratio = 0.25)
            bsz = int((hash_code.size()[0])/2)
            batch=bsz

            if bsz >1:
                u1, u2 = torch.split(hash_code, [bsz, bsz], dim=0)

            features = torch.cat([u1.unsqueeze(1), u2.unsqueeze(1)], dim=1)
            if bsz ==1:
                u1 = hash_code
                u2= hash_code
            
            loss1 = criterion(hash_code, labels.float(), Index,config)
            Loss2 = criterion_contrastive(features, labels.float(),batch)
            train_loss += loss1.item()
            train_loss_contrastive += Loss2.item()
            mixed_loss =  0.8 * loss1 + 0.2 * Loss2
            train_mixed_loss += mixed_loss.item() 
            mixed_loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = train_loss / len(train_loader)
        train_loss_contrastive = train_loss_contrastive / len(train_loader)
        train_mixed_loss = train_mixed_loss / len(train_loader)
        epoch_loss = running_loss / len(train_loader)
        # print(f"Epoch [{epoch+1}/{config['epoch']}], Loss: {epoch_loss:.4f}, Loss: {train_loss:.4f},Loss: {train_loss_contrastive:.4f}")


        print("\b\b\b\b\b\b\b Mixed loss:%.5f train_loss: %.5f Contrastive_loss %.5f recon+loss: %.5f" %
              (train_mixed_loss, train_loss,train_loss_contrastive, epoch_loss))
        # f.write('Train | Epoch: %d | Mixed loss:%.5f | loss:%.5f | contloss:%.5f\n' % (
        #     epoch, train_mixed_loss, train_loss, loss))
        
        if (epoch) % config["test_map"] == 0:
           
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
    
class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=10, contrast_mode='all',
                 base_temperature=10):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels,batch, mask=None):

        if (batch  == 1):
            print("batch..", batch, "features",features)

        device = (torch.device('cuda:1')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')

        features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]*2

        if labels is not None:
            
            labels = (labels == 1).nonzero().squeeze()

            labels = labels[:, 1]
            
            labels = (labels.reshape(labels.size()[-1], 1)).T

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature.T,contrast_feature),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        
        
        if int(labels.shape[0]) == 8:
            mask = mask.repeat(anchor_count*2, contrast_count*2)

        

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
 
        loss= loss.mean()/100

        return loss

if __name__ == "__main__":
    config = get_config()
    config["pr_curve_path"] = f"log/mae/MASKED_{config['dataset']}_{64}.json"
    train_val(config)
