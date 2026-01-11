"""
Training Script for Dynacard Image Classifier
Trains a CNN/ResNet model to classify pump faults from Dynacard images.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.image_classifier import DynacardCNN, DynacardResNet
from utils.data_loader import DYNACARD_DIR, PROJECT_ROOT


class DynacardDataset(Dataset):
    """PyTorch Dataset for Dynacard images."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        self.images = []
        self.labels = []
        self.classes = []
        
        # Load all images from subdirectories
        for class_dir in sorted(self.root_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.classes:
                    self.classes.append(class_name)
                
                class_idx = self.classes.index(class_name)
                
                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                        self.images.append(img_path)
                        self.labels.append(class_idx)
        
        print(f"Loaded {len(self.images)} images across {len(self.classes)} classes")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(train=True, img_size=224):
    """Get image transforms for training/validation."""
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training", leave=False):
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
    
    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    return epoch_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    accuracy = 100. * correct / total
    return epoch_loss, accuracy


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training Loss')
    
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.set_title('Training Accuracy')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Dynacard Classifier")
    parser.add_argument('--data-dir', type=str, default=str(DYNACARD_DIR),
                        help='Path to Dynacard images directory')
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['cnn', 'resnet'], help='Model architecture')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224, help='Image size')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--output-dir', type=str, 
                        default=str(PROJECT_ROOT / 'models' / 'trained'),
                        help='Output directory for saved models')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if data exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("❌ No data found! Please run:")
        print("   python utils/data_loader.py --download")
        return
    
    # Create datasets
    train_transform = get_transforms(train=True, img_size=args.img_size)
    val_transform = get_transforms(train=False, img_size=args.img_size)
    
    full_dataset = DynacardDataset(data_dir, transform=train_transform)
    num_classes = len(full_dataset.classes)
    
    # Split dataset
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Update val transform
    val_dataset.dataset.transform = val_transform
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    print(f"📊 Training samples: {train_size}, Validation samples: {val_size}")
    
    # Create model
    if args.model == 'resnet':
        model = DynacardResNet(num_classes=num_classes, pretrained=True)
        print("🔧 Using ResNet18 with transfer learning")
    else:
        model = DynacardCNN(num_classes=num_classes)
        print("🔧 Using custom CNN")
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      patience=5, factor=0.5)
    
    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                            optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = output_dir / 'best_dynacard_classifier.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': full_dataset.classes,
                'val_acc': val_acc,
                'epoch': epoch
            }, model_path)
            print(f"   ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Plot training history
    plot_path = output_dir / 'training_history.png'
    plot_training_history(train_losses, val_losses, train_accs, val_accs, plot_path)
    
    # Save final model
    final_path = output_dir / 'final_dynacard_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': full_dataset.classes,
        'val_acc': val_acc,
        'epoch': args.epochs
    }, final_path)
    
    print(f"\n🎉 Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.2f}%")
    print(f"   Models saved to: {output_dir}")


if __name__ == "__main__":
    main()
