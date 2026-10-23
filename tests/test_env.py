import sys

def verify_environment():
    report = {}
    print("=" * 60)
    print("PHASE 1: ENVIRONMENT & HARDWARE ACCELERATION VERIFICATION")
    print("=" * 60)

    # 1. Python Version
    print(f"[x] Python Version: {sys.version.split()[0]}")

    # 2. Check Core Dependencies
    modules = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("transformers", "HuggingFace Transformers"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-Learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("multipart", "Python-Multipart"),
        ("pydantic", "Pydantic"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("streamlit", "Streamlit")
    ]

    all_passed = True
    for mod, label in modules:
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "installed")
            print(f"  [+] {label:<26}: OK (v{ver})")
        except ImportError as e:
            print(f"  [-] {label:<26}: FAILED ({e})")
            all_passed = False

    # 3. Check PyTorch & Device Acceleration (CUDA / CPU Fallback)
    print("-" * 60)
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            print(f"  [+] Hardware Acceleration: CUDA ENABLED (GPU)")
            print(f"  [+] Device Name          : {device_name}")
            print(f"  [+] GPU Count            : {device_count}")
            print(f"  [+] CUDA Version         : {cuda_version}")
            
            # Quick tensor compute test on GPU
            x = torch.randn(100, 100, device="cuda")
            y = torch.matmul(x, x)
            print(f"  [+] GPU Tensor Compute   : SUCCESS (Device: {y.device})")
        else:
            print("  [!] Hardware Acceleration: CUDA NOT AVAILABLE (Gracefully falling back to CPU)")
            x = torch.randn(100, 100, device="cpu")
            y = torch.matmul(x, x)
            print(f"  [+] CPU Tensor Compute   : SUCCESS (Device: {y.device})")
    except Exception as e:
        print(f"  [-] Device Verification Failed: {e}")
        all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL REQUIRED PACKAGES AND RUNTIMES VERIFIED SUCCESSFULLY!")
    else:
        print("WARNING: Some packages failed verification.")
    print("=" * 60)
    return all_passed

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
