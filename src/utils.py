import torch
from tqdm import tqdm

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(data_loader, desc="[Val]")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def predict(model, data_loader, device):
    """
    Runs inference on data_loader and returns predictions.
    
    Args:
        model (nn.Module): The model.
        data_loader (DataLoader): Test data.
        device (torch.device): Device.
        
    Returns:
        results (list of tuples): [(filename, class_index), ...]
    """
    model.eval()
    results = []
    
    # We assume data_loader.dataset is SequentialSampler or shuffle=False so ordering matches
    # But for ImageFolder, we can get filepath
    
    samples = data_loader.dataset.samples # list of (path, label)
    # Note: For test set (ImageFolder with dummy classes or flat moved to one class), 
    # label corresponds to subfolder index.
    
    paths = [s[0] for s in samples]
    
    # Check if dataloader is shuffled
    if isinstance(data_loader.sampler, torch.utils.data.RandomSampler):
        print("Warning: Prediction called on shuffled dataloader. Mapping to files might be wrong.")
        
    idx = 0
    with torch.no_grad():
        loop = tqdm(data_loader, desc="[Test]")
        for images, _ in loop:
            images = images.to(device)
            output = model(images)
            _, predicted = output.max(1)
            
            preds = predicted.cpu().tolist()
            
            for p in preds:
                if idx < len(paths):
                    results.append((paths[idx], p))
                    idx += 1
                    
    return results

def measure_class_accuracy(model, data_loader, device, target_class_idx):
    """
    Measures accuracy for a specific class.
    
    Args:
        model (nn.Module): The model.
        data_loader (DataLoader): Validation/Test data.
        device (torch.device): Device.
        target_class_idx (int): The index of the class to evaluate.
        
    Returns:
        accuracy (float): Accuracy for the target class.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        loop = tqdm(data_loader, desc=f"Evaluating Class {target_class_idx}")
        for images, labels in loop:
            # Filter for target class
            mask = labels == target_class_idx
            if mask.sum() == 0:
                continue
                
            images = images[mask].to(device)
            labels = labels[mask].to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if total > 0:
                loop.set_postfix(acc=100.*correct/total)
            
    if total == 0:
        print(f"No samples found for class {target_class_idx}")
        return 0.0
        
    return 100. * correct / total
