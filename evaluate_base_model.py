import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from src.models import get_model


def parse_classes(classes_str):
    if classes_str is None or classes_str.strip() == "":
        return None
    return [int(x) for x in classes_str.split(",") if x.strip() != ""]


def get_cifar100_class_indices(dataset, target_classes):
    indices = []
    for i, label in enumerate(dataset.targets):
        if label in target_classes:
            indices.append(i)
    return indices


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    target_classes = parse_classes(args.classes)
    if target_classes is not None:
        print(f"Evaluating subset classes: {target_classes}")
    else:
        print("Evaluating full CIFAR-100 validation set.")

    # Load data
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_set = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=val_transform)
    if target_classes is None:
        eval_set = val_set
    else:
        bad = [c for c in target_classes if c < 0 or c >= 100]
        if bad:
            raise ValueError(f"Class indices out of range [0, 99]: {bad}")
        val_indices = get_cifar100_class_indices(val_set, target_classes)
        eval_set = Subset(val_set, val_indices)

    val_loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Load Model
    # Important: Base weights are from timm, not X-Pruner models
    model = get_model(args.model_name, pretrained=False, num_classes=100)
    
    print(f"Loading weights from {args.weights_path}")
    state_dict = torch.load(args.weights_path, map_location='cpu')
    # Filter out classifier weights if needed, but since it's finetuned for CIFAR100, we keep it.
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    if target_classes is None:
        print(f"Accuracy on full CIFAR-100 validation set: {acc:.2f}%")
    else:
        print(f"Accuracy on CIFAR-100 subset ({len(target_classes)} classes): {acc:.2f}%")
    print(f"Samples evaluated: {total}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate baseline model on full CIFAR-100 or class subset.")
    parser.add_argument("--weights-path", type=str, default="./weights/deit_tiny_patch16_224_cifar100_finetuned.pth")
    parser.add_argument("--model-name", type=str, default="deit_tiny_patch16_224")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated class indices for subset evaluation.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    evaluate(parser.parse_args())
