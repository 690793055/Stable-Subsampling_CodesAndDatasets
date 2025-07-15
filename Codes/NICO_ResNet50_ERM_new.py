# %%
# % Project Name: USSP
# % Description: Standard ResNet-50 Model 
# % Author: Yang Jinjing
# % Email: yangjinjing94@163.com
# % Date: 2025-07-10
# %%


import os
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from tqdm import tqdm

torch.manual_seed(42)

# --------- Configuration ---------

pkl_dir = '.'
batch_size = 128 #
num_epochs = 10
num_classes = 60
lr = 1e-4
weight_decay = 1e-4
val_split = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class PKLDataset(Dataset):
    def __init__(self, pkl_path, indices=None, val_split=0.1, train=True):
        with open(pkl_path, 'rb') as f:
            full = pickle.load(f)

        if indices is not None:
            indices = np.array(indices).flatten()
            assert (indices >= 0).all() and (indices < len(full)).all(), "Indices out of range"
            all_full_indices = np.arange(len(full))
            remaining_indices = np.setdiff1d(all_full_indices, indices)

            if len(remaining_indices) == 0:
             
                generator = torch.Generator().manual_seed(42) 
                n_val = int(len(full) * val_split)
                n_train = len(full) - n_val
                train_ds, val_ds = random_split(full, [n_train, n_val], generator=generator)
                self.ds = train_ds if train else val_ds
            else:
                if train:
                    self.ds = Subset(full, indices)
                else:
                    val_size_from_remaining = int(len(indices) * val_split)
                    if val_size_from_remaining > len(remaining_indices):
                        val_size_from_remaining = len(remaining_indices)
                    if val_size_from_remaining > 0:
                        np.random.seed(42) 
                        val_idx = np.random.choice(remaining_indices, val_size_from_remaining, replace=False)
                        self.ds = Subset(full, val_idx)
                    else:
                        self.ds = Subset(full, [])
        else:
            generator = torch.Generator().manual_seed(42) 
            n_val = int(len(full) * val_split)
            n_train = len(full) - n_val
            train_ds, val_ds = random_split(full, [n_train, n_val], generator=generator)
            self.ds = train_ds if train else val_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]

class CombinedDataset(Dataset):
    """
    Combine the dataset into a big dataset and return in(image, label, group_idx) form
    """
    def __init__(self, pkl_files, pkl_dir, indices_map, val_split, train=True):
        self.samples = []
        for pkl_file in pkl_files:
            path = os.path.join(pkl_dir, pkl_file)
            current_indices = indices_map.get(pkl_file)
            subset_dataset = PKLDataset(path, current_indices, val_split, train=train)
            for sample in subset_dataset:
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# --------- ERM Model Definition ---------
class ERMResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        for name, param in self.backbone.named_parameters():
            # name.startswith('layer4')   or
            if not (name.startswith('layer4')  or name.startswith('fc')):
                param.requires_grad = False  
            else: 
                param.requires_grad = True
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)


def train_model(model_path, train_data, train_index):
    """
    ERM training

    Args:
        model_path (str): model_path
        train_data (list): train data pkl file
        train_index (list): the index list, if empty, use the full data
    """
    m = len(train_data)
    is_index_provided = bool(train_index)
    if is_index_provided and len(train_data) != len(train_index):
        raise ValueError("train_data and train_index length not eauql")
    env_indices_map = {train_data[i]: (train_index[i] if is_index_provided else None) for i in range(m)}


    train_dataset = CombinedDataset(train_data, pkl_dir, env_indices_map, val_split, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loaders = {}
    for pkl_file in train_data:
        path = os.path.join(pkl_dir, pkl_file)
        current_indices = env_indices_map.get(pkl_file)
        name = pkl_file.replace('.pkl', '')
        val_loaders[name] = DataLoader(
            PKLDataset(path, current_indices, val_split, train=False),
            batch_size=batch_size, shuffle=False, num_workers=0
        )

    model = ERMResNet(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    best_mean_acc = 0.0
    log = {'train_loss': [], 'mean_val_acc': []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100, leave=False)
        total_loss_epoch = 0.0

        for batch in progress:
          
            imgs, labels = batch 
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            total_loss_epoch += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss_epoch / len(train_loader) if len(train_loader) > 0 else 0.0
        

        model.eval()
        val_accs = [evaluate(model, loader, device) for loader in val_loaders.values() if len(loader) > 0]
        mean_val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0

        log['train_loss'].append(avg_loss)
        log['mean_val_acc'].append(mean_val_acc)

        if mean_val_acc > best_mean_acc:
            best_mean_acc = mean_val_acc
            torch.save(model.state_dict(), model_path)
            
        print(
            f"Epoch {epoch}/{num_epochs} Summary: "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Mean Val Acc: {mean_val_acc:.2f}%"
        )

    return best_mean_acc, log



def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            pred = outs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100. * correct / total if total > 0 else 0.0

# --------- Test Scenarios Function (通用 ERM 版本) ---------
def test_scenarios(model_path, test_data):
    """
    Test function

    Args:
        model_path (str): model path
        test_data (list): test data pkl list

    Returns:
        dict: seniro:accuracy
    """
    model = ERMResNet(num_classes).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_results = {}
    print(f"--- Running tests on ERM model: {model_path} ---")

    for pkl_file in test_data:
        path = os.path.join(pkl_dir, pkl_file)
        if not os.path.exists(path):
        
            continue
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            loader = DataLoader(data, batch_size=256, shuffle=False, num_workers=0)
            accuracy = evaluate(model, loader, device)
            scenario_name = pkl_file.replace('.pkl', '')
            test_results[scenario_name] = accuracy
            print(f"  Accuracy on '{pkl_file}': {accuracy:.2f}%")
        except Exception as e:
            test_results[pkl_file.replace('.pkl', '')] = None
            
    print("--- Testing complete ---")
    return test_results