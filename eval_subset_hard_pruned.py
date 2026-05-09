import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from src.models import get_model

def get_cifar100_class_indices(dataset, target_classes):
    indices = []
    for i, label in enumerate(dataset.targets):
        if label in target_classes:
            indices.append(i)
    return indices

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 5 classes
    target_classes = [81, 14, 3, 94, 35]

    # Load data
    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    val_indices = get_cifar100_class_indices(val_set, target_classes)
    val_subset = Subset(val_set, val_indices)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=4)

    # Load Hard-Pruned Model
    model = get_model('deit_tiny_patch16_224', pretrained=False, num_classes=100)
    
    weights_path = './results/test_export/hard_pruned_model.pth'
    print(f"Loading hard-pruned weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
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
    print(f"Accuracy of hard-pruned model on the 5-class subset: {acc:.2f}%")

if __name__ == '__main__':
    evaluate()
