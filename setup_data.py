"""
Direct Download Script for Datasets
Run this after manually downloading the required ZIP files.
"""

import os
import zipfile
import shutil
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
DYNACARD_DIR = DATA_DIR / "dynacard_images"
SENSOR_DIR = DATA_DIR / "sensor_data"
SIMULATOR_DIR = DATA_DIR / "simulator"

# Default download location (Windows)
DOWNLOADS_DIR = Path.home() / "Downloads"


def extract_github_simulator():
    """Extract the GitHub PU simulator from Downloads."""
    zip_name = "PU-master.zip"
    zip_path = DOWNLOADS_DIR / zip_name
    
    if not zip_path.exists():
        print(f"❌ {zip_name} not found in Downloads folder.")
        print(f"   Please download from: https://github.com/vkopey/PU")
        print(f"   Click Code → Download ZIP")
        return False
    
    print(f"📦 Extracting {zip_name}...")
    SIMULATOR_DIR.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(SIMULATOR_DIR)
    
    # Move contents from PU-master subfolder
    extracted = SIMULATOR_DIR / "PU-master"
    if extracted.exists():
        for item in extracted.iterdir():
            target = SIMULATOR_DIR / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(SIMULATOR_DIR))
        extracted.rmdir()
    
    print(f"✅ Simulator extracted to: {SIMULATOR_DIR}")
    return True


def extract_kaggle_sensor():
    """Extract Kaggle pump sensor data from Downloads."""
    possible_names = ["pump-sensor-data.zip", "archive.zip", "sensor.zip"]
    
    zip_path = None
    for name in possible_names:
        path = DOWNLOADS_DIR / name
        if path.exists():
            zip_path = path
            break
    
    if not zip_path:
        print("❌ Pump sensor data ZIP not found in Downloads.")
        print("   Please download from: https://www.kaggle.com/datasets/nphantawee/pump-sensor-data")
        print("   (Kaggle login required)")
        return False
    
    print(f"📦 Extracting {zip_path.name}...")
    SENSOR_DIR.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(SENSOR_DIR)
    
    print(f"✅ Sensor data extracted to: {SENSOR_DIR}")
    return True


def extract_kaggle_images():
    """Extract Kaggle sucker rod images from Downloads."""
    possible_names = ["sucker-rod-image.zip", "archive.zip", "sucker_rod_image.zip"]
    
    zip_path = None
    for name in possible_names:
        path = DOWNLOADS_DIR / name
        if path.exists():
            zip_path = path
            break
    
    if not zip_path:
        print("❌ Sucker rod image ZIP not found in Downloads.")
        print("   Please download from: https://www.kaggle.com/datasets/bairuigong/sucker-rod-image")
        print("   (Kaggle login required)")
        return False
    
    print(f"📦 Extracting {zip_path.name}...")
    DYNACARD_DIR.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DYNACARD_DIR)
    
    print(f"✅ Images extracted to: {DYNACARD_DIR}")
    return True


def check_data_status():
    """Check what data is available."""
    print("\n📊 Data Status:")
    print("-" * 40)
    
    # Check simulator
    if SIMULATOR_DIR.exists() and any(SIMULATOR_DIR.iterdir()):
        py_files = list(SIMULATOR_DIR.glob("*.py"))
        print(f"✅ Simulator: {len(py_files)} Python files")
    else:
        print("❌ Simulator: Not downloaded")
    
    # Check sensor data
    if SENSOR_DIR.exists():
        csv_files = list(SENSOR_DIR.glob("*.csv"))
        if csv_files:
            print(f"✅ Sensor data: {len(csv_files)} CSV files")
        else:
            print("❌ Sensor data: Empty")
    else:
        print("❌ Sensor data: Not downloaded")
    
    # Check image data
    if DYNACARD_DIR.exists():
        subdirs = [d for d in DYNACARD_DIR.iterdir() if d.is_dir()]
        if subdirs:
            total_images = sum(len(list(d.glob("*.png")) + list(d.glob("*.jpg"))) for d in subdirs)
            print(f"✅ Dynacard images: {total_images} images in {len(subdirs)} classes")
        else:
            print("❌ Dynacard images: Empty")
    else:
        print("❌ Dynacard images: Not downloaded")


if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("  AUTONOMOUS PUMP - DATA SETUP SCRIPT")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--extract":
        print("\n🔄 Extracting all available downloads...")
        extract_github_simulator()
        extract_kaggle_sensor()
        extract_kaggle_images()
    
    check_data_status()
    
    print("\n📝 Manual Download Instructions:")
    print("-" * 40)
    print("1. GitHub Simulator (No login required):")
    print("   → https://github.com/vkopey/PU")
    print("   → Click 'Code' → 'Download ZIP'")
    print()
    print("2. Kaggle Sensor Data (Login required):")
    print("   → https://www.kaggle.com/datasets/nphantawee/pump-sensor-data")
    print("   → Click 'Download' button")
    print()
    print("3. Kaggle Dynacard Images (Login required):")
    print("   → https://www.kaggle.com/datasets/bairuigong/sucker-rod-image")
    print("   → Click 'Download' button")
    print()
    print("After downloading, run: python setup_data.py --extract")
