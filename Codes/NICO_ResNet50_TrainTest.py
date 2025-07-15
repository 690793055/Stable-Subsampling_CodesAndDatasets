"""
% Project Name: USSP
% Description: The ResNet-50 model train and test
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
"""


from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, Subset, ConcatDataset
import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import pickle
import random


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        torch.manual_seed(11)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Global configuration
train_dir = "../realdata/NICO/public_dg_0416/train/autumn"  # Path to the training set
test_dirs = [                                         # List of test scenarios
    "../realdata/NICO/public_dg_0416/train/dim",
    "../realdata/NICO/public_dg_0416/train/grass",
    "../realdata/NICO/public_dg_0416/train/outdoor",
    "../realdata/NICO/public_dg_0416/train/rock",
    "../realdata/NICO/public_dg_0416/train/water"
]
batch_size = 64
num_epochs = 1
num_classes = 60  # Adjust according to the actual number of classes

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading function
def load_datasets(dataset,subsample_index):
    # filename = "NICO_autumn_enhanced_dataset.pkl"

    # try:
    #     with open(filename, 'rb') as f:
    #         full_train_set = pickle.load(f)

    #     print(f"Loaded dataset size: {len(full_train_set)}")

    # except FileNotFoundError:
    #     print(f"Error: File {filename} not found.")
    # except Exception as e:
    #     print(f"Error reading file: {e}")
    full_train_set=dataset


    # Load and validate subsample indices
    subsample_indices = subsample_index.flatten()
    assert (subsample_indices >= 0).all() and (subsample_indices < len(full_train_set)).all(), "Indices out of valid range"

    if len(subsample_indices) < len(full_train_set):
        # Subsample indices are not the entire dataset

        # Create subsampled training set
        train_subset = Subset(full_train_set, subsample_indices)

        # Find the indices of the remaining data
        all_indices = np.arange(len(full_train_set))
        remaining_indices = np.setdiff1d(all_indices, subsample_indices)

        if len(remaining_indices) > 0:
            # Determine the validation set size (1/10 of the training set)
            val_size = int(len(train_subset) * 0.1)
            if val_size > len(remaining_indices):
                val_size = len(remaining_indices) # Validation set size does not exceed the remaining data

            if val_size > 0:
                # Randomly select indices from the remaining data as the validation set
                np.random.seed(42) # Set random seed for reproducibility
                val_indices = np.random.choice(remaining_indices, val_size, replace=False)
                val_subset = Subset(full_train_set, val_indices)
            else:
                val_subset = Subset(full_train_set, []) # Create an empty validation set if there is not enough remaining data
        else:
            val_subset = Subset(full_train_set, []) # Create an empty validation set if there is no remaining data

        return train_subset, val_subset

    else:
        # Subsample indices are the entire dataset, keep the original validation set split logic
        subsampled_dataset = Subset(full_train_set, subsample_indices)
        train_size = int(0.95 * len(subsampled_dataset))
        val_size = len(subsampled_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(subsampled_dataset, [train_size, val_size])
        return train_subset, val_subset

# Model initialization
def create_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Freezing strategy: last two layers are trainable
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc.requires_grad = True

    return model.to(device)


# Model training
def train_model(dataset, subsample_indices, save_path="best_model.pth", batch_size=32, num_epochs=20,lr=0.0001):
    # Initialize training logger
    train_log = {'loss': [], 'val_acc': []}

    # Load data
    train_set, val_set = load_datasets(dataset, subsample_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)  # lr=0.0001 currently the best for testing

    best_acc = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0  # Accumulate loss

        # Wrap train_loader with tqdm and add description
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",  # Progress bar title
            ncols=100,  # Progress bar width
            leave=True   # Keep the progress bar visible
        )

        for images, labels in train_progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Update progress bar information in real-time
            epoch_loss += loss.item()
            train_progress.set_postfix(loss=f"{loss.item():.4f}") # Show the loss of the current batch

        # Record training loss
        train_log['loss'].append(epoch_loss / len(train_loader))

        # Validate and save the best model
        val_acc = evaluate(model, val_loader)
        train_log['val_acc'].append(val_acc)
        # print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss / len(train_loader):.4f}, Validation Accuracy: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    return best_acc, train_log

# Evaluation function (reused)
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Multi-scenario testing function
# def test_scenarios(model_path="best_model.pth"):
#     model = create_model()
#     model.load_state_dict(torch.load(model_path))

#     test_results = {}
#     seed=11
#     for scenario_path in test_dirs:

#         scenario_name = os.path.basename(scenario_path.rstrip('/'))
#         # Load test set
#         test_set = datasets.ImageFolder(
#             root=scenario_path,
#             transform= transforms.Compose([
#                                         transforms.Resize((224, 224)),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#                                         ])
#         )
#         test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)

#         # Perform evaluation
#         acc_original = evaluate(model, test_loader)
#         test_results[f"{scenario_name}_original"] = acc_original

#         # Create transform to add Gaussian noise
#         min_std = 0.1
#         max_std = 0.3

#         random.seed(seed)
#         random_std = random.uniform(min_std, max_std)
#         gaussian_noise_transform = AddGaussianNoise(0., random_std)

#         # Create new test set and apply noise transform
#         noisy_test_set = datasets.ImageFolder(
#             root=scenario_path,
#             transform=transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 gaussian_noise_transform,
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#             ])
#         )
#         noisy_test_loader = DataLoader(noisy_test_set, batch_size=128, shuffle=False, num_workers=0)

#         # Perform evaluation on the test set with added Gaussian noise
#         acc_noisy = evaluate(model, noisy_test_loader)
#         test_results[f"{scenario_name}_noisy_std_{random_std:.2f}"] = acc_noisy
#         seed+=1


#     return test_results

def test_scenarios(model_path="best_model.pth"):
    model = create_model()
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set model to evaluation mode

    test_results = {}
    preprocessed_dir = "preprocessed_test_data"
    processed_scenarios = set()

    if not os.path.exists(preprocessed_dir):
        print(f"Warning: Preprocessed data directory not found at {preprocessed_dir}. Please run preprocess_test_data first.")
        return test_results

    for filename in os.listdir(preprocessed_dir):
        if filename.endswith(".pkl"):
            if "_original.pkl" in filename:
                scenario_name = filename.replace("_original.pkl", "")
                if scenario_name not in processed_scenarios:
                    pkl_path_original = os.path.join(preprocessed_dir, filename)
                    pkl_path_noisy = os.path.join(preprocessed_dir, f"{scenario_name}_noisy.pkl")

                    if os.path.exists(pkl_path_original):
                        try:
                            with open(pkl_path_original, 'rb') as f:
                                original_data = pickle.load(f)
                            original_loader = DataLoader(original_data, batch_size=256, shuffle=False, num_workers=0)
                            acc_original = evaluate(model, original_loader)
                            test_results[f"{scenario_name}_original"] = acc_original
                        except Exception as e:
                            print(f"Error loading or evaluating original data for {scenario_name}: {e}")
                            test_results[f"{scenario_name}_original"] = None
                    else:
                        print(f"Warning: Preprocessed original data not found for {scenario_name} at {pkl_path_original}.")
                        test_results[f"{scenario_name}_original"] = None

                    if os.path.exists(pkl_path_noisy):
                        try:
                            with open(pkl_path_noisy, 'rb') as f:
                                noisy_data = pickle.load(f)
                            noisy_loader = DataLoader(noisy_data, batch_size=128, shuffle=False, num_workers=0)
                            acc_noisy = evaluate(model, noisy_loader)
                            test_results[f"{scenario_name}_noisy"] = acc_noisy
                        except Exception as e:
                            print(f"Error loading or evaluating noisy data for {scenario_name}: {e}")
                            test_results[f"{scenario_name}_noisy"] = None
                    else:
                        print(f"Warning: Preprocessed noisy data not found for {scenario_name} at {pkl_path_noisy}.")
                        test_results[f"{scenario_name}_noisy"] = None

                    processed_scenarios.add(scenario_name)

    return test_results