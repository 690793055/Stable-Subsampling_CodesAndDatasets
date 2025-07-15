# %%
# % Project Name: USSP
# % Description: IRM ResNet-50 Model 
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
from torch.autograd import grad
from torchvision import models
from torch.utils.data import DataLoader, Subset, Dataset,random_split
from tqdm import tqdm


torch.manual_seed(42)

# --------- Configuration ---------

pkl_dir = '.'
batch_size = 128 
num_epochs = 15
num_classes = 60
lr = 1e-4
lambda_irm = 1
weight_decay = 1e-4
val_split = 0.1 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------- Dataset Wrapper with Subsampling ---------
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


# --------- IRM Model Definition ---------
class IRMResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        for name, param in self.backbone.named_parameters():
            # 
            if not (name.startswith('layer4')  or name.startswith('fc')):     
                param.requires_grad = False  
            else:
                param.requires_grad = True
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return self.backbone(x) * self.scale


def train_model(model_path, train_data, train_index=[]):
    """
    IRM training

    Args:
        model_path (str): model_path
        train_data (list): train data pkl file
        train_index (list): the index list, if empty, use the full data
    """
    train_loaders = {}
    val_loaders = {}
    is_index_provided = bool(train_index) 
    if is_index_provided and len(train_data) != len(train_index):
        raise ValueError("train_data and train_index length not eauql")
    for i, pkl_file in enumerate(train_data):
        path = os.path.join(pkl_dir, pkl_file)
        
        current_indices = train_index[i] if is_index_provided else None

        train_ds = PKLDataset(path, current_indices, val_split, train=True)
        val_ds = PKLDataset(path, current_indices, val_split, train=False)

        name = pkl_file.replace('.pkl', '')
        train_loaders[name] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loaders[name] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
    
    model = IRMResNet(num_classes).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    log = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        
        train_iters = [iter(loader) for loader in train_loaders.values()]

        min_len = min(len(loader) for loader in train_loaders.values())

        total_loss_epoch = 0.0
        
        progress = tqdm(
            range(min_len),
            desc=f"Epoch {epoch}/{num_epochs}",
            ncols=100, 
            leave=False 
        )
        
        for i in progress:
            optimizer.zero_grad()
            
            total_erm_loss = 0.0
            total_penalty = 0.0
            
            for env_iter in train_iters:
                imgs, labels = next(env_iter)
                imgs, labels = imgs.to(device), labels.to(device)
                
                outputs = model(imgs)
                loss_env = criterion(outputs, labels)

                total_erm_loss += loss_env
                
                grad_s = grad(loss_env, model.scale, create_graph=True)[0]
                total_penalty += torch.mean(grad_s ** 2)

            avg_erm_loss = total_erm_loss / len(train_loaders)
            avg_penalty = total_penalty / len(train_loaders)
            
            final_loss = avg_erm_loss + lambda_irm * avg_penalty
            
            final_loss.backward()
            optimizer.step()
            
            total_loss_epoch += final_loss.item()
            progress.set_postfix(loss=f"{final_loss.item():.4f}")

        avg_loss = total_loss_epoch / min_len if min_len > 0 else 0
        
        if not val_loaders:
            mean_val_acc = 0.0
        else:
            val_accs = [evaluate(model, loader, device) for loader in val_loaders.values() if len(loader) > 0]
            mean_val_acc = sum(val_accs) / len(val_accs) if val_accs else 0.0

        log['train_loss'].append(avg_loss)
        log['val_acc'].append(mean_val_acc)

        if mean_val_acc > best_val_acc:
            best_val_acc = mean_val_acc
            torch.save(model.state_dict(), model_path)

        summary_text = (
            f"Epoch {epoch}/{num_epochs} Summary: "
            f"Avg Loss: {avg_loss:.4f}, "
            f"Val Acc: {mean_val_acc:.2f}%"
        )
        print(summary_text)

    return best_val_acc, log


# --------- Evaluate Function ---------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# --------- Test Scenarios Function (修改后) ---------
def test_scenarios(model_path, test_data):
    """
    Test function

    Args:
        model_path (str): model path
        test_data (list): test data pkl list

    Returns:
        dict: seniro:accuracy
    """
    model = IRMResNet(num_classes).to(device) 
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_results = {}
    print(f"--- Running tests on model: {model_path} ---")

    for pkl_file in test_data:
        path = os.path.join(pkl_dir, pkl_file)

        if not os.path.exists(path):
            continue

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            loader = DataLoader(
                data,
                batch_size=256,  
                shuffle=False,
                num_workers=0
            )

            accuracy = evaluate(model, loader, device)
            
            scenario_name = pkl_file.replace('.pkl', '')
            test_results[scenario_name] = accuracy
            print(f"  Accuracy on '{pkl_file}': {accuracy:.2f}%")

        except Exception as e:
            print(f"Processing '{path}' error: {e}")
            scenario_name = pkl_file.replace('.pkl', '')
            test_results[scenario_name] = None
            
    print("--- Testing complete ---")
    return test_results