import torch

def get_best_device():
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Using NVIDIA GPU:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple MPS GPU")
    else:
        # Try DirectML for AMD GPUs if installed
        try:
            import torch_directml
            device = torch_directml.device()
            print("🔥 Using AMD GPU via DirectML")
        except ImportError:
            device = torch.device("cpu")
            print("💻 Using CPU (no compatible GPU detected)")
    return device