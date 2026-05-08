import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image


def get_dataloaders(data_dir='./data', dataset_name='cifar10', batch_size=32, image_size=224, num_workers=2, train=True, split='val'):
    """
    Creates DataLoaders for CIFAR-10, CIFAR-100, or ImageNet.
    
    Args:
        data_dir (str): Directory where data is stored.
        dataset_name (str): 'cifar10', 'cifar100', or 'imagenet'.
        batch_size (int): Batch size.
        image_size (int): Input image size.
        num_workers (int): Number of dataloader workers.
        
    Returns:
        train_loader, val_loader
    """
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)), # Keep resize if strict on size, but usually RandomResizedCrop is better
        # However, for fine-tuning on small images (CIFAR), sometimes simple resize+crop is enough.
        # Let's use a strong recipe:
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)), # Less aggressive scale for fine-tuning? or standard (0.08, 1.0)
        # Let's stick to standard RandomResizedCrop but maybe slightly less aggressive since we fine-tune
        # transforms.RandomResizedCrop(image_size), 
        # Actually, user wants improvement. Let's use TrivialAugment or RandAugment
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load datasets
    if dataset_name.lower() == 'cifar10':
        if train:
            train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=train_transform
            )
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform
        )
    elif dataset_name.lower() == 'cifar100':
        if train:
            train_dataset = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=train_transform
            )
        val_dataset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=val_transform
        )
    elif dataset_name.lower() == 'imagenet':
        # Expects structure:
        # data_dir/train/class_x/xxx.jpg
        # data_dir/val/class_x/xxx.jpg
        train_dir = f"{data_dir}/train"
        val_dir = f"{data_dir}/val"
        if train:
            train_dataset = torchvision.datasets.ImageFolder(
                root=train_dir, transform=train_transform
            )
        val_dataset = torchvision.datasets.ImageFolder(
            root=val_dir, transform=val_transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if split == 'test':
        train_loader = None
        if dataset_name.lower() == 'imagenet':
            test_dir = f"{data_dir}/test"
            # Helper: Check if flat or structured
            # ImageFolder requires structure. We assume ./data/imagenet/test/unknown/xxx.jpg
            # If not, we might fail.
            test_dataset = torchvision.datasets.ImageFolder(
                root=test_dir, transform=val_transform
            )
            val_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            return train_loader, val_loader
        else:
             # Cifar test is usually val? Or distinct? CIFAR10/100 has train=False as test.
             # We just return the standard val_loader logic but maybe distinct?
             # For CIFAR, 'val' usually serves as test.
             pass

    train_loader = None
    if train:
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader

class SingleClassDataset(Dataset):
    def __init__(self, root_dir, class_index, transform=None):
        """
        Args:
            root_dir (str): Path to the validation directory (containing class folders).
            class_index (int): The index of the class to load (0-999).
            transform (callable, optional): Transform to apply.
        """
        self.root_dir = root_dir
        self.class_index = class_index
        self.transform = transform
        
        # Get sorted list of directories to map index to folder name
        dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if class_index < 0 or class_index >= len(dirs):
            raise ValueError(f"Class index {class_index} out of range (0-{len(dirs)-1})")
            
        self.class_folder = dirs[class_index]
        self.class_path = os.path.join(root_dir, self.class_folder)
        
        self.image_files = sorted([f for f in os.listdir(self.class_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.class_path, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, self.class_index

def get_single_class_dataloader(data_dir, dataset_name, target_class, batch_size=32, image_size=224, num_workers=2):
    """
    Creates a DataLoader for a single class without scanning the entire dataset.
    """
    if dataset_name.lower() != 'imagenet':
        raise NotImplementedError("Single class optimization currently only for ImageNet")
        
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_dir = f"{data_dir}/val"
    dataset = SingleClassDataset(val_dir, target_class, transform=val_transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
