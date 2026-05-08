import timm
import torch
import torch.nn as nn
import os

def get_model(model_name, num_classes=10, pretrained=True, weights_dir='./weights'):
    """
    Creates a model using timm, loading/saving weights locally.
    
    Args:
        model_name (str): Name of the model.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.
        weights_dir (str): Directory to store weights.
        
    Returns:
        model (nn.Module): The PyTorch model.
    """
    try:
        weights_path = os.path.join(weights_dir, f"{model_name}.pth")
        
        if os.path.exists(weights_path) and pretrained:
            print(f"Loading local weights from {weights_path}")
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
            # Load the state dict. Note: strict=False might be needed if heads differ, 
            # but usually for same num_classes it's fine. 
            # If the saved weights were for 1000 classes (ImageNet) and we want 10 or 100, 
            # we should handle that.
            # However, typically 'pretrained=True' in timm downloads ImageNet weights. 
            # If we want to Cache that, we should download with pretrained=True once.
            
            # Better approach for caching timm weights:
            # timm handles caching automatically in ~/.cache/huggingface/hub or similar.
            # But user asked to "store the models locally".
            
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle potential mismatch in head if we are finetuning
            # For simplicity, let's assume if we load local weights, they fit.
            # Or we can load with strict=False
            model.load_state_dict(state_dict, strict=False)
            
        else:
            if pretrained:
                print(f"Downloading/Loading pretrained weights for {model_name}...")
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            
            if pretrained:
                print(f"Saving weights to {weights_path}")
                os.makedirs(weights_dir, exist_ok=True)
                torch.save(model.state_dict(), weights_path)
                
        return model
    except ValueError as e:
        print(f"Error: Model {model_name} not found in timm. Please check the name.")
        raise e
