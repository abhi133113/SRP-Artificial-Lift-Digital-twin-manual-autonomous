"""
Training Script for Sensor Anomaly Detector
Trains an LSTM Autoencoder to detect pump failures from sensor data.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.anomaly_detector import LSTMAutoencoder, AnomalyDetector
from utils.data_loader import load_sensor_dataset, SENSOR_DIR, PROJECT_ROOT
from utils.preprocessing import normalize_sensor_data, create_sequences


class SensorDataset(Dataset):
    """PyTorch Dataset for sensor sequences."""
    
    def __init__(self, sequences, labels=None):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.sequences[idx], self.labels[idx]
        return self.sequences[idx]


def train_epoch(model, loader, criterion, optimizer, device):
    """Train autoencoder for one epoch."""
    model.train()
    running_loss = 0.0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        if isinstance(batch, (list, tuple)):
            sequences = batch[0].to(device)
        else:
            sequences = batch.to(device)
        
        optimizer.zero_grad()
        reconstructed = model(sequences)
        loss = criterion(reconstructed, sequences)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * sequences.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """Validate the autoencoder."""
    model.eval()
    running_loss = 0.0
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            if isinstance(batch, (list, tuple)):
                sequences = batch[0].to(device)
            else:
                sequences = batch.to(device)
            
            reconstructed = model(sequences)
            loss = criterion(reconstructed, sequences)
            
            running_loss += loss.item() * sequences.size(0)
            
            # Per-sample reconstruction error
            mse = torch.mean((sequences - reconstructed) ** 2, dim=(1, 2))
            all_errors.extend(mse.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(all_errors)


def plot_training_history(train_losses, val_losses, save_path):
    """Plot and save training loss history."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.legend()
    plt.title('Anomaly Detector Training')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training plot saved to {save_path}")


def plot_reconstruction_errors(errors, threshold, save_path):
    """Plot reconstruction error distribution."""
    plt.figure(figsize=(10, 4))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Error distribution saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Sensor Anomaly Detector")
    parser.add_argument('--data-dir', type=str, default=str(SENSOR_DIR),
                        help='Path to sensor data directory')
    parser.add_argument('--seq-length', type=int, default=50, 
                        help='Sequence length for LSTM')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64, help='LSTM hidden dim')
    parser.add_argument('--latent-dim', type=int, default=32, help='Latent space dim')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--threshold-percentile', type=float, default=95,
                        help='Percentile for anomaly threshold')
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
    
    # Load data
    try:
        df, labels = load_sensor_dataset(data_dir=args.data_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please run: python utils/data_loader.py --download")
        return
    
    # Filter to normal data only if labels available (for training)
    if labels is not None:
        normal_mask = labels.str.upper() == 'NORMAL'
        df_normal = df[normal_mask].copy()
        print(f"Using {len(df_normal)} NORMAL samples for training")
    else:
        df_normal = df.copy()
    
    # Normalize data
    df_norm = normalize_sensor_data(df_normal, method='zscore')
    
    # Handle missing values
    df_norm = df_norm.fillna(df_norm.mean())
    df_norm = df_norm.replace([np.inf, -np.inf], 0)
    
    # Create sequences
    data = df_norm.values.astype(np.float32)
    sequences = create_sequences(data, seq_length=args.seq_length, stride=1)
    print(f"Created {len(sequences)} sequences of length {args.seq_length}")
    
    input_dim = sequences.shape[2]
    print(f"Input dimension: {input_dim} features")
    
    # Create dataset and split
    dataset = SensorDataset(sequences)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    print(f"📊 Training sequences: {train_size}, Validation sequences: {val_size}")
    
    # Create model
    model = LSTMAutoencoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=2
    )
    model = model.to(device)
    print(f"🔧 Model: LSTM Autoencoder (hidden={args.hidden_dim}, latent={args.latent_dim})")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                      patience=5, factor=0.5)
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_errors = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_errors = val_errors
    
    # Compute threshold from validation errors
    threshold = np.percentile(best_errors, args.threshold_percentile)
    print(f"\n📏 Anomaly threshold (P{args.threshold_percentile}): {threshold:.6f}")
    
    # Save model with threshold
    model_path = output_dir / 'anomaly_detector.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'seq_length': args.seq_length,
        'threshold': threshold,
        'val_loss': best_val_loss
    }, model_path)
    print(f"✅ Model saved to {model_path}")
    
    # Plot training history
    plot_path = output_dir / 'anomaly_training_history.png'
    plot_training_history(train_losses, val_losses, plot_path)
    
    # Plot error distribution
    error_plot_path = output_dir / 'reconstruction_errors.png'
    plot_reconstruction_errors(best_errors, threshold, error_plot_path)
    
    print(f"\n🎉 Training complete!")
    print(f"   Best validation loss: {best_val_loss:.6f}")
    print(f"   Anomaly threshold: {threshold:.6f}")


if __name__ == "__main__":
    main()
