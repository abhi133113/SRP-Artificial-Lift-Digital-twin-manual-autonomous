"""
Prediction/Inference Script for Autonomous Pump
Run inference on images or sensor data using trained models.
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.image_classifier import DynacardResNet
from models.anomaly_detector import LSTMAutoencoder
from utils.preprocessing import preprocess_dynacard_image, normalize_sensor_data, create_sequences
from utils.data_loader import PROJECT_ROOT


def load_classifier(model_path):
    """Load trained Dynacard classifier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    classes = checkpoint['classes']
    
    model = DynacardResNet(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, classes, device


def load_anomaly_detector(model_path):
    """Load trained anomaly detector."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = LSTMAutoencoder(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        latent_dim=checkpoint['latent_dim'],
        num_layers=2
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    threshold = checkpoint.get('threshold', None)
    seq_length = checkpoint.get('seq_length', 50)
    
    return model, threshold, seq_length, device


def predict_image(image_path, model_path=None):
    """
    Predict fault type from a Dynacard image.
    
    Args:
        image_path: Path to image file
        model_path: Path to trained model (default: models/trained/best_dynacard_classifier.pth)
    
    Returns:
        dict with prediction results
    """
    if model_path is None:
        model_path = PROJECT_ROOT / 'models' / 'trained' / 'best_dynacard_classifier.pth'
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with train_classifier.py")
    
    model, classes, device = load_classifier(model_path)
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    img_tensor = preprocess_dynacard_image(image)
    img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    predicted_class = classes[pred_idx.item()]
    confidence = confidence.item()
    
    # All probabilities
    all_probs = {classes[i]: probs[0, i].item() for i in range(len(classes))}
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': all_probs,
        'image_path': str(image_path)
    }


def predict_sensor_anomaly(data, model_path=None):
    """
    Detect anomalies in sensor data.
    
    Args:
        data: numpy array or pandas DataFrame of sensor readings
        model_path: Path to trained model
    
    Returns:
        dict with anomaly detection results
    """
    if model_path is None:
        model_path = PROJECT_ROOT / 'models' / 'trained' / 'anomaly_detector.pth'
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with train_anomaly.py")
    
    model, threshold, seq_length, device = load_anomaly_detector(model_path)
    
    # Convert DataFrame to numpy if needed
    if isinstance(data, pd.DataFrame):
        # Drop entirely NaN columns to match training data
        all_nan_cols = data.columns[data.isna().all()].tolist()
        if all_nan_cols:
            data = data.drop(columns=all_nan_cols)
        data = data.select_dtypes(include=[np.number]).values
    
    # Normalize
    data = data.astype(np.float32)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std == 0] = 1
    data_norm = (data - mean) / std
    
    # Handle NaN/Inf
    data_norm = np.nan_to_num(data_norm, nan=0, posinf=0, neginf=0)
    
    # Create sequences
    if len(data_norm) < seq_length:
        raise ValueError(f"Data too short. Need at least {seq_length} samples.")
    
    sequences = create_sequences(data_norm, seq_length, stride=seq_length)
    seq_tensor = torch.FloatTensor(sequences).to(device)
    
    # Compute reconstruction errors
    model.eval()
    with torch.no_grad():
        reconstructed = model(seq_tensor)
        errors = torch.mean((seq_tensor - reconstructed) ** 2, dim=(1, 2))
        errors = errors.cpu().numpy()
    
    # Detect anomalies
    if threshold is not None:
        is_anomaly = errors > threshold
    else:
        is_anomaly = np.zeros(len(errors), dtype=bool)
    
    # Health score (0-100)
    if threshold:
        health_scores = np.clip(100 * (1 - errors / (2 * threshold)), 0, 100)
    else:
        health_scores = np.ones(len(errors)) * 50
    
    return {
        'num_sequences': len(sequences),
        'anomalies_detected': int(np.sum(is_anomaly)),
        'anomaly_rate': float(np.mean(is_anomaly)),
        'avg_health_score': float(np.mean(health_scores)),
        'reconstruction_errors': errors.tolist(),
        'is_anomaly': is_anomaly.tolist(),
        'threshold': threshold
    }


def main():
    parser = argparse.ArgumentParser(description="Pump Fault Prediction")
    
    subparsers = parser.add_subparsers(dest='command', help='Prediction type')
    
    # Image prediction
    img_parser = subparsers.add_parser('image', help='Classify Dynacard image')
    img_parser.add_argument('image_path', type=str, help='Path to Dynacard image')
    img_parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    
    # Sensor prediction
    sensor_parser = subparsers.add_parser('sensor', help='Detect sensor anomalies')
    sensor_parser.add_argument('csv_path', type=str, help='Path to sensor CSV file')
    sensor_parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.command == 'image':
        print(f"🔍 Analyzing Dynacard image: {args.image_path}")
        result = predict_image(args.image_path, args.model)
        
        print(f"\n✅ Prediction: {result['prediction']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print("\n   All probabilities:")
        for fault, prob in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
            bar = '█' * int(prob * 20)
            print(f"   {fault:15s} {prob:.2%} {bar}")
    
    elif args.command == 'sensor':
        print(f"🔍 Analyzing sensor data: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        result = predict_sensor_anomaly(df, args.model)
        
        print(f"\n📊 Analysis Results:")
        print(f"   Sequences analyzed: {result['num_sequences']}")
        print(f"   Anomalies detected: {result['anomalies_detected']}")
        print(f"   Anomaly rate: {result['anomaly_rate']:.2%}")
        print(f"   Avg health score: {result['avg_health_score']:.1f}/100")
        
        if result['anomalies_detected'] > 0:
            print("\n   ⚠️  WARNING: Anomalies detected! Check pump status.")
        else:
            print("\n   ✅ All sequences appear normal.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
