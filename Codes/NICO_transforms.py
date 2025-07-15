"""
% Project Name: USSP
% Description: The ehanced trian dataset of NICO-autumn
% Author: Yang Jinjing
% Email: yangjinjing94@163.com
% Date: 2025-04-19
"""


import torch
import torchvision.transforms as transforms
import numpy as np
import random
import torchvision.datasets as datasets
from PIL import Image

# Custom Gaussian noise class
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        torch.manual_seed(11)
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# List of transformations
transform_list = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3,0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    # transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomRotation(30),
    #     #transforms.ColorJitter(0.2, 0.2, 0.2,0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ]),
    # transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     #transforms.ColorJitter(0.3, 0.3, 0.3,0.2),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(0.2, 0.2, 0.2,0.3),
        transforms.ToTensor(),
        AddGaussianNoise(0., 0.2),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def NICO_transforms(image):
    """Applies the list of transformations and returns the list of transformed images.

    Args:
        image (PIL.Image): The original image.
        transform_list (list): The list of transformations.

    Returns:
        list: The list of transformed images.
    """
    transformed_images = []
    for transform in transform_list:
        set_seed(11)
        transformed_images.append(transform(image))
    return transformed_images