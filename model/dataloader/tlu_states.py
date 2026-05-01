import torch
import os.path as osp
import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

def identity(x):
    return x

class TLUStates(Dataset):
    def __init__(self, setname, args, augment=False):
        self.setname = setname
        self.args = args
        self.image_path = args.image_path
        self.split_json = args.split_json
        
        if self.image_path is None or self.split_json is None:
            raise ValueError("image_path and split_json must be provided for TLUStates dataset")

        with open(self.split_json, 'r') as f:
            self.split = json.load(f)

        self.data, self.label = self.parse_data()
        self.num_class = len(set(self.label))

        image_size = 84
        if augment and setname == 'train':
            transforms_list = [
                  transforms.RandomResizedCrop(image_size),
                  transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                ]
        else:
            transforms_list = [
                  transforms.Resize(92),
                  transforms.CenterCrop(image_size),
                  transforms.ToTensor(),
                ]

        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422,  115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272,   68.27635443,  72.54505529]]))
            ])
        else:
            # Default ResNet/WRN normalization
            self.transform = transforms.Compose(
                transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

    def parse_data(self):
        classes = self.split[self.setname]
        data = []
        label = []
        
        # We need a consistent mapping of class names to labels across all sets if they share classes.
        # But in FSL, train/val/test splits usually have disjoint classes.
        # Here we just map them relative to the current split to keep it simple, 
        # as the Sampler uses labels to group images of the same class.
        
        for lb, class_name in enumerate(classes):
            class_dir = osp.join(self.image_path, class_name.strip())
            if not osp.exists(class_dir):
                print(f"Warning: Class directory {class_dir} does not exist.")
                continue
                
            filenames = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for filename in filenames:
                data.append(osp.join(class_dir, filename))
                label.append(lb)
                
        return data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
