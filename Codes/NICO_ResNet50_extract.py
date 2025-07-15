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

# 固定随机种子
torch.manual_seed(42)

# --------- Configuration ---------
pkl_dir = '.'
batch_size = 128
num_epochs = 10
num_classes = 60
lr = 1e-4
weight_decay = 1e-4
val_split = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --------- Dataset Wrappers (无需改动) ---------
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
                print(f"警告: 'indices' 包含了 '{pkl_path}' 的所有样本，将对全部数据进行随机划分。")
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

# --------- ERM Model Definition (修改后) ---------
class ERMResNet(nn.Module):
    def __init__(self, num_classes, dim_extract):
        """
        模型初始化。
        
        Args:
            num_classes (int): 最终分类任务的类别数。
            dim_extract (int): 中间特征提取的维度。
        """
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        
        # 冻结除最后几层外的所有层
        for name, param in self.backbone.named_parameters():
            if not (name.startswith('layer4') or name.startswith('fc')):
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        # 1. 将原始的全连接层 'fc' 替换为一个新的线性层，用于特征提取
        # 输出维度为 dim_extract
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            dim_extract
        )

        # 2. 新增一个独立的分类器层，用于在训练时进行分类
        # 输入维度是 dim_extract，输出维度是 num_classes
        self.classifier = nn.Linear(dim_extract, num_classes)

    def forward(self, x):
        """
        用于训练和评估的前向传播。
        返回最终的分类 logits。
        """
        features = self.backbone(x) # 提取特征
        outputs = self.classifier(features) # 进行分类
        return outputs
        
    def extract_features(self, x):
        """
        专门用于特征提取的方法。
        返回指定维度的特征向量。
        """
        return self.backbone(x)


# --------- Train Function (修改后) ---------
def train_model(model_path, train_data, train_index, dim_extract):
    """
    通用 ERM 训练函数，并在训练后提取特征。

    Args:
        model_path (str): 保存最佳模型的路径。
        train_data (list): 用于训练的 pkl 文件名列表。
        train_index (list): 对应的索引列表。
        dim_extract (int): 需要提取的特征维度。
    """
    m = len(train_data)
    is_index_provided = bool(train_index)
    if is_index_provided and len(train_data) != len(train_index):
        raise ValueError("train_data 和 train_index 的长度必须一致！")

    env_indices_map = {train_data[i]: (train_index[i] if is_index_provided else None) for i in range(m)}

    train_dataset = CombinedDataset(train_data, pkl_dir, env_indices_map, val_split, train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loaders = {}
    for pkl_file in train_data:
        path = os.path.join(pkl_dir, pkl_file)
        current_indices = env_indices_map.get(pkl_file)
        val_loaders[pkl_file] = DataLoader(
            PKLDataset(path, current_indices, val_split, train=False),
            batch_size=batch_size, shuffle=False, num_workers=0
        )

    # 在模型实例化时传入 dim_extract
    model = ERMResNet(num_classes=num_classes, dim_extract=dim_extract).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_mean_acc = 0.0
    log = {'train_loss': [], 'mean_val_acc': []}
    
    # --- 训练阶段 ---
    print(f"--- Starting Training for {num_epochs} epochs ---")
    for epoch in range(1, num_epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", ncols=100, leave=True)
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
        print(f"Epoch {epoch}/{num_epochs} Summary: Avg Loss: {avg_loss:.4f}, Mean Val Acc: {mean_val_acc:.2f}%")
    
    print(f"--- Training finished. Best model saved to {model_path} with Mean Val Acc: {best_mean_acc:.2f}% ---")
    
    # --- 特征提取阶段 ---
    print(f"\n--- Starting Feature Extraction (dim={dim_extract}) ---")
    # 加载性能最佳的模型
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for i, pkl_file in enumerate(train_data):
            path = os.path.join(pkl_dir, pkl_file)
            current_indices = env_indices_map.get(pkl_file)
            
            # 为当前文件创建一个新的数据集，确保包含所有用于训练的样本
            # 注意：这里的 val_split=0, train=True 确保了 PKLDataset 返回包含所有指定索引的完整训练子集
            extract_dataset = PKLDataset(path, current_indices, val_split=0, train=True)
            extract_loader = DataLoader(extract_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            
            print(f"Extracting from '{pkl_file}' ({len(extract_dataset)} samples)...")
            
            features_list = []
            for imgs, _ in tqdm(extract_loader, desc=f"  Extracting {pkl_file}", leave=False):
                imgs = imgs.to(device)
                # 使用专门的 extract_features 方法
                features = model.extract_features(imgs)
                features_list.append(features.cpu().numpy())
            
            if not features_list:
                print(f"  No data to extract from '{pkl_file}'.")
                continue

            # 合并所有批次的特征
            all_features = np.vstack(features_list)
            
            # 定义并保存 .npy 文件
            output_npy_path = os.path.join(pkl_dir, pkl_file.replace('.pkl', f'_features_{dim_extract}d.npy'))
            np.save(output_npy_path, all_features)
            print(f"  Features saved to '{output_npy_path}' with shape {all_features.shape}")
            
    print("--- Feature Extraction complete ---")

    return best_mean_acc, log


# --------- Evaluate Function (无需改动) ---------
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

# test_scenarios 函数已被移除