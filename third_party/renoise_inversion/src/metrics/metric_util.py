import torch
from PIL import Image
from torchvision import transforms

from src.metrics.lpips import LPIPS
import torch.nn as nn

dev = 'cuda'
to_tensor_transform = transforms.Compose([transforms.ToTensor()])
mse_loss = nn.MSELoss()

def calculate_l2_difference(image1, image2, device: str = None): # Changed default
    # Determine target device if not provided
    if device is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            target_device = "xpu"
        elif torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"
    else:
        target_device = device
        
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)

    mse = mse_loss(image1, image2).item()
    return mse

def calculate_psnr(image1, image2, device: str = None): # Changed default
    # Determine target device if not provided
    if device is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            target_device = "xpu"
        elif torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"
    else:
        target_device = device
        
    max_value = 1.0
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)
    
    mse = mse_loss(image1, image2)
    psnr = 10 * torch.log10(max_value**2 / mse).item()
    return psnr


# loss_fn = LPIPS(net_type='vgg').to(dev).eval() # Commented out: Initialize inside function for device safety

def calculate_lpips(image1, image2, device: str = None): # Changed default
    """Calculates LPIPS distance, initializing model on the target device."""
    # Determine target device if not provided
    if device is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            target_device = "xpu"
        elif torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"
    else:
        target_device = device
        
    # Initialize LPIPS model on the target device within the function
    # Assuming LPIPS is imported correctly at the top of the file
    try:
        # Use a local variable for the LPIPS model instance
        loss_fn_local = LPIPS(net_type='vgg').to(target_device).eval()
    except NameError:
        print("[calculate_lpips] ERROR: LPIPS class not found. Ensure 'lpips' library is installed and imported.")
        return None # Or raise an error
        
    if isinstance(image1, Image.Image):
        # Assuming to_tensor_transform is defined elsewhere
        image1 = to_tensor_transform(image1).to(target_device)
    if isinstance(image2, Image.Image):
        # Assuming to_tensor_transform is defined elsewhere
        image2 = to_tensor_transform(image2).to(target_device)
    
    # Calculate loss using the locally initialized model
    with torch.no_grad(): # LPIPS calculation should not require gradients
        loss = loss_fn_local(image1, image2).item()
    return loss

def calculate_metrics(image1, image2, device: str = None, size=(512, 512)): # Changed default
    """Calculates L2, PSNR, and LPIPS metrics, auto-detecting device."""
    # Determine target device if not provided
    if device is None:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            target_device = "xpu"
        elif torch.cuda.is_available():
            target_device = "cuda"
        else:
            target_device = "cpu"
    else:
        target_device = device
        
    if isinstance(image1, Image.Image):
        image1 = image1.resize(size)
        # Assuming to_tensor_transform is defined elsewhere
        image1 = to_tensor_transform(image1).to(target_device)
    if isinstance(image2, Image.Image):
        image2 = image2.resize(size)
        # Assuming to_tensor_transform is defined elsewhere
        image2 = to_tensor_transform(image2).to(target_device)
        
    # Pass the determined device to the other metric functions
    l2 = calculate_l2_difference(image1, image2, target_device)
    psnr = calculate_psnr(image1, image2, target_device)
    lpips = calculate_lpips(image1, image2, target_device)
    
    # Return None for metrics if lpips calculation failed (returned None)
    if lpips is None:
        print("[calculate_metrics] Warning: LPIPS calculation failed, returning None for metrics.")
        return None
        
    return {"l2": l2, "psnr": psnr, "lpips": lpips}

def get_empty_metrics():
    return {"l2": 0, "psnr": 0, "lpips": 0}

def print_results(results):
    print(f"Reconstruction Metrics: L2: {results['l2']},\t PSNR: {results['psnr']},\t LPIPS: {results['lpips']}")