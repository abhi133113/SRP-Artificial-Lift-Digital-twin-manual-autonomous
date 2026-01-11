# SRP Digital Twin: Autonomous Pump Control System

## 1. Context: The "Flying Blind" Problem
Sucker Rod Pumps (SRPs) are the heartbeat of onshore oil production, yet operators face critical challenges:
-   **Lack of Visibility**: Operators cannot see downhole conditions in real-time. They rely on "Surface Cards" which are distorted by rod stretch and friction.
-   **Reactive Maintenance**: Faults like "Fluid Pound" (pump hitting fluid hard) are often detected only *after* equipment failure.
-   **Inefficient Operations**: Pumps run at fixed speeds. If the well dries up, the pump keeps running, wasting energy and damaging rods.

## 2. The Solution: An Autonomous Digital Twin
We have built a system that acts as the **"Brain"** of the pump. It doesn't just monitor; it **optimizes**.

### Core Philosophy: "The 3-Second Rule"
The UI is designed so an operator can understand the well's health in **3 seconds or less**. It is organized into three zones:

---

## 3. UI Guide: The "Ultimate Dashboard"

### ZONE 1: THE PULSE (Left Panel) – Real-Time Physics
*   **What it is**: The raw physical data stream.
*   **Component: Dual Dynacard Chart**
    *   **Green Line (Surface Card)**: The raw load measured at the polished rod. This is "Physical Truth".
    *   **Red Line (Downhole Card)**: The calculated load at the pump, derived using the **Wave Equation**. This is "Calculated Truth".
    *   **Why**: Engineers trust the Surface card but *need* the Downhole card to diagnose the pump. Showing both proves the physics engine is accurate.
*   **Component: Virtual Pump Animation**
    *   **What**: A visual representation of the rod moving up and down.
    *   **How**: Synchronized 1:1 with the live **Strokes Per Minute (SPM)**.
    *   **Why**: Instant visual confirmation that the system is live and connected.

### ZONE 2: THE BRAIN (Center Panel) – AI & Insights
*   **What it is**: The intelligence layer. It converts data into decisions.
*   **Component: Pump Fillage Gauge**
    *   **What**: Shows how full the pump chamber is (0-100%).
    *   **Ghost Needle**: A grey marker showing the **Target Setpoint** (e.g., 90%).
    *   **Why**: Fillage is the #1 KPI for efficiency. Low fillage = Fluid Pound (Damage). High fillage = Missed Production.
*   **Component: Goodman Diagram**
    *   **What**: A structural engineering chart plotting **Min Stress** vs **Max Stress**.
    *   **Logic**: If the dot is inside the triangle, the rod is **Safe**. If outside, **Fatigue Failure** is imminent.
    *   **Why**: This speaks the language of mechanical engineers. It predicts rod snaps *before* they happen.

### ZONE 3: THE HANDS (Right Panel) – Control & Transparency
*   **What it is**: The action layer. It builds trust in automation.
*   **Component: Explainable AI (XAI) Log**
    *   **What**: A feed of the AI's decisions.
    *   **The Twist**: It explains **WHY**.
    *   *Example*: "Fillage < 70% -> Reducing Speed to prevent Fluid Pound."
    *   **Why**: Operators won't trust a "Black Box". They need to know the reason behind every action.
*   **Component: Shadow Mode Toggle**
    *   **Advisory Mode**: AI suggests actions, human approves.
    *   **Autonomous Mode**: AI acts alone.
    *   **Why**: Allows safe testing of the system without risking the well.
*   **Component: Signal Confidence (SNR)**
    *   **What**: A WiFi-style signal bar.
    *   **How**: Calculates Signal-to-Noise Ratio from the sensors.
    *   **Why**: Admits when data is noisy, which builds credibility.

### ZONE 4: THE MISSION BRIEF (Overlay) – System Validation
*   **What it is**: A "Click to Reveal" deep dive into the system's architecture.
*   **Component: Validation Matrix**
    *   **What**: A table proving exactly how we solve the core problems (Manual Interpretation, Fluid Pound, Rod Fatigue).
*   **Component: The "Secret Sauce"**
    *   **What**: Highlights the shift from **Detection** (Passive) to **Correction** (Active).
*   **Component: Feature Importance**
    *   **What**: A SHAP-style chart showing which variables (Dynacard Shape, Fillage, etc.) drive the AI's decisions.

---

## 4. Technical Stack
-   **Backend**: Python (FastAPI) for physics simulation and AI inference.
-   **Frontend**: HTML5/CSS3 (Glassmorphism) + Chart.js for visualization.
-   **AI Models**:
    -   **ResNet18**: For Dynacard Image Classification (Visual Diagnosis).
    -   **LSTM Autoencoder**: For Sensor Anomaly Detection (Time-Series).
-   **Communication**: WebSockets for sub-50ms real-time updates.

## 5. How to Run
1.  **Start the System**:
    ```bash
    python server.py
    ```
2.  **Access the Dashboard**:
    Open `http://localhost:8005` in your browser.
