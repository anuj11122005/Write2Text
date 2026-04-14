"""
=============================================================================
Check Setup -- Verify Everything is Ready for Training
=============================================================================
Run this script to verify your environment is properly configured.

Usage:
    python check_setup.py
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if all required packages are installed."""
    print("  Checking dependencies...")
    required = {
        "torch": "torch",
        "torchvision": "torchvision",
        "pandas": "pandas",
        "numpy": "numpy",
        "cv2": "opencv-python",
        "sklearn": "scikit-learn",
        "matplotlib": "matplotlib",
        "tqdm": "tqdm",
    }

    all_ok = True
    for module_name, pip_name in required.items():
        try:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "?")
            print(f"    [OK]   {pip_name:<20s} {version}")
        except ImportError:
            print(f"    [FAIL] {pip_name:<20s} NOT INSTALLED")
            all_ok = False

    return all_ok


def check_gpu():
    """Check GPU availability."""
    print("\n  Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"    [OK]   GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print(f"    [WARN] No GPU detected -- will use CPU (slower training)")
            return True
    except Exception as e:
        print(f"    [FAIL] GPU check failed: {e}")
        return True  # Not a fatal error


def check_dataset():
    """Check if dataset files exist."""
    print("\n  Checking dataset...")
    from src import config

    words_txt = os.path.join(config.DATASET_RAW, "words.txt")
    words_dir = os.path.join(config.DATASET_RAW, "words")

    all_ok = True

    if os.path.exists(words_txt):
        # Count lines
        with open(words_txt, "r", encoding="utf-8", errors="replace") as f:
            lines = sum(1 for line in f if not line.startswith("#") and line.strip())
        print(f"    [OK]   words.txt found ({lines:,} entries)")
    else:
        print(f"    [FAIL] words.txt NOT FOUND at: {words_txt}")
        all_ok = False

    if os.path.isdir(words_dir):
        # Count a few images
        img_count = 0
        for root, dirs, files in os.walk(words_dir):
            img_count += sum(1 for f in files if f.endswith(".png"))
            if img_count > 1000:
                break
        print(f"    [OK]   words/ directory found ({img_count}+ images)")
    else:
        print(f"    [FAIL] words/ directory NOT FOUND at: {words_dir}")
        all_ok = False

    return all_ok


def check_processed_data():
    """Check if processed data exists."""
    print("\n  Checking processed data...")
    from src import config

    files_to_check = [
        ("train.csv", "Training data"),
        ("val.csv", "Validation data"),
        ("test.csv", "Test data"),
        ("char_mapping.json", "Character mapping"),
    ]

    all_ok = True
    for filename, description in files_to_check:
        path = os.path.join(config.DATASET_PROCESSED, filename)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"    [OK]   {filename:<25s} ({size_kb:.1f} KB)")
        else:
            print(f"    [WARN] {filename:<25s} not found (run: python scripts/prepare_data.py)")
            all_ok = False

    return all_ok


def check_model():
    """Check if model can be instantiated."""
    print("\n  Checking model...")
    try:
        import torch
        from src.model import CRNN
        from src import config

        model = CRNN(num_classes=80)  # Dummy number
        dummy = torch.randn(2, 1, config.IMG_HEIGHT, config.IMG_WIDTH)
        with torch.no_grad():
            output = model(dummy)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"    [OK]   CRNN model instantiated")
        print(f"    [OK]   Input:  {tuple(dummy.shape)}")
        print(f"    [OK]   Output: {tuple(output.shape)}")
        print(f"    [OK]   Params: {total_params:,}")
        return True
    except Exception as e:
        print(f"    [FAIL] Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_checkpoint():
    """Check if trained model exists."""
    print("\n  Checking checkpoints...")
    from src import config

    ckpt_path = os.path.join(config.CHECKPOINTS_DIR, "best_model.pth")
    if os.path.exists(ckpt_path):
        size_mb = os.path.getsize(ckpt_path) / 1e6
        print(f"    [OK]   best_model.pth found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"    [WARN] No trained model found (run: python train.py)")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("  Write2Text -- Setup Verification")
    print("=" * 60)

    results = {}
    results["dependencies"] = check_dependencies()
    results["gpu"] = check_gpu()
    results["dataset_raw"] = check_dataset()
    results["dataset_processed"] = check_processed_data()
    results["model"] = check_model()
    results["checkpoint"] = check_checkpoint()

    # Summary
    print("\n" + "=" * 60)
    print("  Summary:")
    print("=" * 60)

    all_critical_ok = True
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"    {status}  {name}")
        if name in ["dependencies", "model"] and not ok:
            all_critical_ok = False

    if all_critical_ok:
        if not results["dataset_raw"]:
            print("\n  Dataset not found. Steps:")
            print("    1. Download IAM Words dataset")
            print("    2. Place in: dataset/raw/iam_words/")
            print("    3. Run: python scripts/prepare_data.py")
            print("    4. Run: python train.py")
        elif not results["dataset_processed"]:
            print("\n  Data not prepared. Run:")
            print("    python scripts/prepare_data.py")
        elif not results["checkpoint"]:
            print("\n  Ready to train! Run:")
            print("    python train.py")
        else:
            print("\n  Everything is ready!")
            print("    Predict: python predict.py --image <image_path>")
    else:
        print("\n  Critical issues found. Fix them first.")

    print("=" * 60)


if __name__ == "__main__":
    main()
