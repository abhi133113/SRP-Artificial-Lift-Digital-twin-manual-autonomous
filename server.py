import os
import json
import asyncio
import base64
import random
import sys
import math
from pathlib import Path
from datetime import datetime

# Reconfigure stdout/stderr to UTF-8 for Windows compatibility
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from predict import load_classifier, load_anomaly_detector, preprocess_dynacard_image
from utils.data_loader import DYNACARD_DIR, PROJECT_ROOT

app = FastAPI(title="SRP Digital Twin API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
TRAINED_MODELS_DIR = PROJECT_ROOT / "models" / "trained"
CLASSIFIER_PATH = TRAINED_MODELS_DIR / "best_dynacard_classifier.pth"
ANOMALY_PATH = TRAINED_MODELS_DIR / "anomaly_detector.pth"
SENSOR_DATA_PATH = PROJECT_ROOT / "data" / "sensor_data" / "sensor.csv"

# Global state
models = {}
sensor_data_cache = []
history_cache = [] # List of dicts: {timestamp, fillage, stress, spm}
MAX_HISTORY = 100

# Simulation State (Physics-Based)
sim_state = {
    "target_spm": 5.0,
    "current_spm": 5.2,
    "target_fillage": 90.0,  # Ghost needle setpoint
    "strokes_saved": 142,
    "energy_saved_kwh": 12.5,
    "total_strokes": 0,
    "stroke_phase": 0.0,  # Current position in stroke cycle (0-2*pi)
    "mode": "advisory",  # "advisory" or "autonomous"
    # Goodman Diagram: Track stress history
    "min_stress": 0.0,
    "max_stress": 0.0,
}

# ... (rest of file)

@app.get("/api/history")
async def get_history():
    return history_cache

@app.get("/analytics.html")
async def get_analytics():
    return FileResponse("analytics.html")

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/deadlines.html")
async def get_deadlines():
    return FileResponse("deadlines.html")

# ---------------------------------------------------------
# PHYSICS SIMULATION FUNCTIONS
# ---------------------------------------------------------

def generate_dynacard_physics(stroke_phase, fillage):
    """
    Generate Surface and Downhole dynacards.
    Surface Card: Measured at polished rod (includes rod stretch, friction).
    Downhole Card: Calculated via Wave Equation (what the pump actually sees).
    
    Both are plotted as Load (Y) vs Position (X).
    Returns arrays of [x, y] points for each card.
    """
    num_points = 50
    positions = np.linspace(0, 100, num_points)  # 0 to 100% stroke
    
    # Surface Card (Idealized sinusoidal + load effects)
    # Higher fillage = more fluid load = higher peak load
    fluid_load = fillage * 0.5  # 0-50 units based on fillage
    rod_weight = 30  # Constant rod weight
    friction_loss = 5 * np.sin(positions * np.pi / 100)  # Position-dependent friction
    
    surface_load = rod_weight + fluid_load * np.sin(positions * np.pi / 100) + friction_loss
    
    # Add some realistic noise to surface card
    noise = np.random.normal(0, 1, num_points)
    surface_load += noise
    
    # Downhole Card (Wave equation effects - phase shifted, damped)
    # The downhole card "lags" the surface due to wave propagation
    phase_lag = 15  # degrees equivalent
    damping = 0.85  # Signal attenuation
    
    downhole_positions = np.roll(positions, int(phase_lag * num_points / 360))
    downhole_load = damping * (rod_weight + fluid_load * np.sin(downhole_positions * np.pi / 100))
    
    # If fillage is low (fluid pound), downhole card shows characteristic "flat top"
    if fillage < 70:
        # Fluid pound signature: flat section in downstroke
        pound_start = int(num_points * 0.6)
        pound_end = int(num_points * 0.85)
        downhole_load[pound_start:pound_end] = downhole_load[pound_start] * 0.5
    
    # Format as arrays for JSON
    surface_card = [[float(positions[i]), float(surface_load[i])] for i in range(num_points)]
    downhole_card = [[float(positions[i]), float(downhole_load[i])] for i in range(num_points)]
    
    return surface_card, downhole_card

def calculate_goodman_stress(surface_card, spm):
    """
    Calculate Min/Max stress for Goodman Diagram.
    Goodman Diagram plots (Min Stress, Max Stress) to check fatigue safety.
    If point is inside the "safe triangle", the rod is safe.
    """
    loads = [pt[1] for pt in surface_card]
    max_load = max(loads)
    min_load = min(loads)
    
    # Convert load to stress (simplified: stress = load / rod_area)
    rod_area = 1.0  # Normalized
    max_stress = max_load / rod_area
    min_stress = min_load / rod_area
    
    # Fatigue accumulates with speed
    fatigue_factor = spm / 5.0  # Normalized to 5 SPM baseline
    max_stress *= fatigue_factor
    
    # Check if safe (inside Goodman triangle)
    # Simplified: Safe if max_stress < 80 and stress range < 50
    stress_range = max_stress - min_stress
    is_safe = (max_stress < 80) and (stress_range < 50)
    
    return {
        "min_stress": round(min_stress, 2),
        "max_stress": round(max_stress, 2),
        "is_safe": is_safe
    }

def calculate_signal_confidence():
    """
    Calculate Signal-to-Noise Ratio (SNR) for sensor reliability.
    Returns confidence level: "high", "medium", or "low".
    """
    # Simulate SNR based on various factors
    base_snr = 25  # dB
    noise_variation = random.uniform(-5, 5)
    snr = base_snr + noise_variation
    
    if snr > 22:
        return {"level": "high", "snr_db": round(snr, 1), "bars": 4}
    elif snr > 15:
        return {"level": "medium", "snr_db": round(snr, 1), "bars": 2}
    else:
        return {"level": "low", "snr_db": round(snr, 1), "bars": 1}

def calculate_time_to_failure(goodman_data, fillage):
    """
    Calculate estimated Time to Failure (TTF) in hours.
    Based on Stress (Fatigue) and Shock (Fluid Pound).
    """
    # Base life: 5 years (43,800 hours)
    base_life_hours = 43800
    
    # Factor 1: Stress (Goodman)
    # If max_stress > 90% of limit, life drops exponentially
    stress_ratio = goodman_data['max_stress'] / 100.0  # Assuming 100 is limit
    stress_factor = 1.0
    if stress_ratio > 0.8:
        stress_factor = 0.1  # Critical stress reduces life 10x
    elif stress_ratio > 0.6:
        stress_factor = 0.5
        
    # Factor 2: Shock (Fluid Pound)
    # Low fillage = High shock
    shock_factor = 1.0
    if fillage < 60:
        shock_factor = 0.05  # Severe pounding destroys pump in days
    elif fillage < 80:
        shock_factor = 0.2
        
    # Calculate TTF
    ttf_hours = base_life_hours * stress_factor * shock_factor
    
    # Urgency Level
    urgency = "NORMAL"
    if ttf_hours < 24:
        urgency = "CRITICAL"
    elif ttf_hours < 168: # 1 week
        urgency = "MAINTENANCE REQUIRED"
        
    return {
        "hours": round(ttf_hours, 1),
        "urgency": urgency,
        "stress_factor": round(stress_factor, 2),
        "shock_factor": round(shock_factor, 2)
    }

def generate_xai_reason(diagnosis, fillage, action, mode):
    """
    Generate Explainable AI (XAI) reasoning.
    Not just WHAT happened, but WHY.
    """
    reasons = []
    
    if "Fluid Pound" in diagnosis:
        reasons.append(f"Fillage dropped to {fillage:.0f}% indicating incomplete pump fill")
        reasons.append("Gas likely entering pump chamber")
        if action != "Monitoring":
            reasons.append(f"Slowing to allow gas separation and fluid recovery")
    elif "Gas" in diagnosis:
        reasons.append(f"Gas interference detected (Fillage: {fillage:.0f}%)")
        reasons.append("Recommend venting or speed reduction")
    elif "Normal" in diagnosis:
        if fillage > 95:
            reasons.append(f"Pump running full (Fillage: {fillage:.0f}%)")
            reasons.append("Speed increase possible for higher production")
        else:
            reasons.append(f"Operating within optimal range (Fillage: {fillage:.0f}%)")
    
    if mode == "autonomous":
        reasons.append("AUTONOMOUS MODE: AI executing control action")
    else:
        reasons.append("ADVISORY MODE: Awaiting operator confirmation")
    
    return " | ".join(reasons) if reasons else "System nominal"

# ---------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------

def load_all_models():
    print("Loading models...")
    try:
        if CLASSIFIER_PATH.exists():
            models['classifier'], models['classes'], models['device'] = load_classifier(CLASSIFIER_PATH)
            print(f"Image Classifier loaded ({len(models['classes'])} classes)")
        
        if ANOMALY_PATH.exists():
            models['anomaly'], models['threshold'], models['seq_length'], _ = load_anomaly_detector(ANOMALY_PATH)
            print(f"Anomaly Detector loaded (Threshold: {models['threshold']:.4f})")
    except Exception as e:
        print(f"Error loading models: {e}")

def load_sensor_data():
    global sensor_data_cache
    print(f"Loading sensor data from {SENSOR_DATA_PATH}...")
    try:
        if not SENSOR_DATA_PATH.exists():
            print(f"Sensor data not found at {SENSOR_DATA_PATH}")
            return
        
        df = pd.read_csv(SENSOR_DATA_PATH)
        df_clean = df.copy()
        non_numeric = df_clean.select_dtypes(exclude=[np.number]).columns
        df_clean = df_clean.drop(columns=non_numeric)
        all_nan_cols = df_clean.columns[df_clean.isna().all()].tolist()
        df_clean = df_clean.drop(columns=all_nan_cols)
        df_clean = df_clean.fillna(0)
        
        sensor_data_cache = df_clean.to_dict(orient='records')
        print(f"Loaded {len(sensor_data_cache)} sensor rows for streaming")
    except Exception as e:
        print(f"Error loading sensor data: {e}")

@app.on_event("startup")
async def startup_event():
    load_all_models()
    load_sensor_data()

# ---------------------------------------------------------
# DYNACARD IMAGE HELPER
# ---------------------------------------------------------

def get_random_dynacard():
    dataset_dir = DYNACARD_DIR / "dataset"
    if not dataset_dir.exists():
        return None, None
    
    images = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
    if not images:
        return None, None
    
    img_path = random.choice(images)
    try:
        with open(img_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode('utf-8')
        return img_base64, img_path
    except Exception as e:
        print(f"Error reading image {img_path}: {e}")
        return None, None

# ---------------------------------------------------------
# LIVE INFERENCE & CONTROL LOGIC
# ---------------------------------------------------------

def run_live_inference(sensor_row, img_path):
    """
    Main inference function that generates all dashboard data.
    """
    global sim_state, history_cache
    results = {}
    
    # 1. Image Classification & Fillage Simulation
    diagnosis = "Normal Operation"
    fillage = 92.0
    confidence = 0.0
    
    if 'classifier' in models and img_path:
        try:
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess_dynacard_image(img)
            img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).to(models['device'])
            
            with torch.no_grad():
                outputs = models['classifier'](img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf_tensor, pred_idx = torch.max(probs, 1)
                confidence = float(conf_tensor.item())
                
            class_id = models['classes'][pred_idx.item()]
            
            # Map class to industrial diagnosis
            if class_id == '0': 
                diagnosis = "Normal Operation"
                fillage = random.uniform(88, 98)
            elif class_id in ['1', '2', '3']:
                diagnosis = "Gas Interference"
                fillage = random.uniform(60, 75)
            elif class_id in ['4', '5', '6']:
                diagnosis = "Fluid Pound"
                fillage = random.uniform(40, 65)
            else:
                diagnosis = f"Anomaly Type {class_id}"
                fillage = random.uniform(70, 85)
                
            results['classification'] = {
                'diagnosis': diagnosis,
                'confidence': confidence
            }
        except Exception as e:
            print(f"Inference error (image): {e}")
    else:
        # Fallback simulation if no model
        fillage = 85 + 10 * math.sin(sim_state['total_strokes'] / 10)
        diagnosis = "Normal Operation" if fillage > 75 else "Fluid Pound"
        confidence = 0.95
        results['classification'] = {'diagnosis': diagnosis, 'confidence': confidence}

    # 2. Autonomous Control Logic
    action = "Monitoring"
    action_desc = "Maintaining current speed"
    
    if fillage < 70:
        action = "Reducing Speed"
        if sim_state['mode'] == "autonomous":
            sim_state['current_spm'] = max(3.0, sim_state['current_spm'] - 0.1)
        action_desc = f"Fillage low ({fillage:.0f}%) -> Reducing speed to {sim_state['current_spm']:.1f} SPM"
        sim_state['strokes_saved'] += 1
        sim_state['energy_saved_kwh'] += 0.05
    elif fillage > 95 and sim_state['current_spm'] < sim_state['target_spm']:
        action = "Increasing Speed"
        if sim_state['mode'] == "autonomous":
            sim_state['current_spm'] = min(6.0, sim_state['current_spm'] + 0.1)
        action_desc = f"Fillage high ({fillage:.0f}%) -> Increasing speed to {sim_state['current_spm']:.1f} SPM"
    
    results['control'] = {
        'action': action,
        'description': action_desc,
        'target_spm': sim_state['target_spm'],
        'current_spm': round(sim_state['current_spm'], 1),
        'fillage': round(fillage, 1),
        'target_fillage': sim_state['target_fillage'],
        'mode': sim_state['mode']
    }
    
    # 3. Generate Dual Dynacard (Surface + Downhole)
    surface_card, downhole_card = generate_dynacard_physics(sim_state['stroke_phase'], fillage)
    results['dynacard_physics'] = {
        'surface': surface_card,
        'downhole': downhole_card,
        'stroke_position': (math.sin(sim_state['stroke_phase']) + 1) / 2 * 100  # 0-100% for animation
    }
    
    # 4. Goodman Diagram (Structural Safety)
    goodman = calculate_goodman_stress(surface_card, sim_state['current_spm'])
    sim_state['min_stress'] = goodman['min_stress']
    sim_state['max_stress'] = goodman['max_stress']
    results['goodman'] = goodman
    
    # 5. Signal Confidence (SNR)
    results['signal_confidence'] = calculate_signal_confidence()
    
    # 6. Time to Failure (Predictive Maintenance)
    results['ttf'] = calculate_time_to_failure(goodman, fillage)
    
    # 7. Explainable AI (XAI) Reason
    results['xai_reason'] = generate_xai_reason(diagnosis, fillage, action, sim_state['mode'])
    
    # 7. Financial Metrics
    results['financial'] = {
        'strokes_saved': sim_state['strokes_saved'],
        'energy_saved': round(sim_state['energy_saved_kwh'], 2)
    }

    # 8. Motor Amps Simulation
    t = np.linspace(0, 4*np.pi, 50)
    amps = 10 + 5 * np.sin(t + sim_state['stroke_phase']) + np.random.normal(0, 0.3, 50)
    results['amps'] = amps.tolist()
    
    # Update History Cache
    timestamp = datetime.now().strftime("%H:%M:%S")
    history_item = {
        "timestamp": timestamp,
        "fillage": round(fillage, 1),
        "stress": round(goodman['max_stress'], 1),
        "spm": round(sim_state['current_spm'], 1)
    }
    history_cache.append(history_item)
    if len(history_cache) > MAX_HISTORY:
        history_cache.pop(0)
    
    # Update stroke phase for next iteration
    sim_state['stroke_phase'] += 0.3  # ~every 2 seconds moves ~0.3 radians
    sim_state['total_strokes'] += 1
    
    return results

# ---------------------------------------------------------
# WEBSOCKET ENDPOINT
# ---------------------------------------------------------

@app.websocket("/websocket")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket")
    
    if not sensor_data_cache:
        print("No sensor data to stream. Closing connection.")
        await websocket.close()
        return
        
    try:
        idx = 0
        print(f"Starting WebSocket stream with {len(sensor_data_cache)} rows")
        while True:
            row = sensor_data_cache[idx % len(sensor_data_cache)]
            img_b64, img_path = get_random_dynacard()
            inference = run_live_inference(row, img_path)
            
            packet = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "dynacard_image": img_b64,
                "analysis": inference
            }
            
            await websocket.send_json(packet)
            
            idx += 1
            await asyncio.sleep(2)  # Stream every 2 seconds
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# ---------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------

@app.post("/api/set_mode")
async def set_mode(mode: str):
    """Toggle between 'advisory' and 'autonomous' mode."""
    if mode in ["advisory", "autonomous"]:
        sim_state['mode'] = mode
        return {"status": "ok", "mode": mode}
    return {"status": "error", "message": "Invalid mode"}

@app.get("/")
async def get_index():
    return FileResponse("index.html")

@app.get("/deadlines.html")
async def get_deadlines():
    return FileResponse("deadlines.html")

# Serve static files
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8005)
