import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from src.models import get_model
from src.dataset import get_dataloaders
from src.utils import train_one_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description='Train Deit/MobileViT on CIFAR10/100/ImageNet')
    parser.add_argument('--model', type=str, default='deit_tiny_patch16_224', help='Model name (timm)')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100', 'imagenet'], help='Dataset name')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--weights-dir', type=str, default='./weights', help='Directory to store/load weights')
    parser.add_argument('--num-classes', type=int, default=100, help='Number of classes')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='Split to evaluate on')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate only')
    parser.add_argument('--target-class', type=int, default=None, help='Class index to calculate accuracy for')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Adjust num_classes defaults if not explicitly set might be good, but user can pass it.
    
    # Data
    print(f"Loading data: {args.dataset}")
    try:
        if args.target_class is not None and args.dataset == 'imagenet':
            from src.dataset import get_single_class_dataloader
            train_loader = None
            val_loader = get_single_class_dataloader(args.data_dir, args.dataset, args.target_class, args.batch_size)
        else:
            train_loader, val_loader = get_dataloaders(
                data_dir=args.data_dir, 
                dataset_name=args.dataset,
                batch_size=args.batch_size,
                train=not args.eval_only,
                split=args.split
            )
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Model
    print(f"Creating model: {args.model}")
    model = get_model(
        args.model, 
        num_classes=args.num_classes, 
        weights_dir=args.weights_dir
    )
    
    if args.checkpoint:
        print(f"Resuming from checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        
    model.to(device)
    
    # Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4) # Added weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.dry_run:
        print("Dry run initiated...")
        images, labels = next(iter(train_loader if train_loader else val_loader))
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        print(f"Dry run successful. Loss: {loss.item()}")
        return

    if args.eval_only:
        if args.split == 'test':
            print("Starting inference on test set...")
            from src.utils import predict
            import csv
            import os
            
            # Predict
            predictions = predict(model, val_loader, device)
            
            # Save predictions
            save_path = "predictions.csv"
            print(f"Saving predictions to {save_path}...")
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Id', 'Prediction']) # Header
                for path, pred in predictions:
                    filename = os.path.basename(path)
                    writer.writerow([filename, pred])
            print("Done.")
            return
        
        print("Starting evaluation...")
        
        if args.target_class is not None:
            from src.utils import measure_class_accuracy
            acc = measure_class_accuracy(model, val_loader, device, args.target_class)
            print(f"Accuracy for class {args.target_class}: {acc:.2f}%")
        else:    
            val_loss, val_acc = evaluate(model, criterion, val_loader, device)
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        return

    # Training Loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch
        )
        scheduler.step() # Step scheduler
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    print("Training complete.")
    
    # Save the finetuned model
    save_path = f"{args.weights_dir}/{args.model}_{args.dataset}_finetuned.pth"
    print(f"Saving finetuned model to {save_path}")
    torch.save(model.state_dict(), save_path)

if __name__ == '__main__':
    main()
