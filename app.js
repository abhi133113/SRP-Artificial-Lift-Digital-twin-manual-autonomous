// =========================================
// ULTIMATE SRP DIGITAL TWIN - JavaScript
// Physics-Based, Explainable AI Dashboard
// =========================================

// DOM Elements
const connectionStatus = document.getElementById('connectionStatus');
const systemStatus = document.getElementById('systemStatus');
const signalIndicator = document.getElementById('signalIndicator');
const snrValue = document.getElementById('snrValue');
const latencyEl = document.getElementById('latency');
const lastUpdated = document.getElementById('lastUpdated');
const aiDiagnosis = document.getElementById('aiDiagnosis');
const confidenceFill = document.getElementById('confidenceFill');
const xaiReason = document.getElementById('xaiReason');
const fillageValue = document.getElementById('fillageValue');
const actionLog = document.getElementById('actionLog');
const strokesSaved = document.getElementById('strokesSaved');
const energySaved = document.getElementById('energySaved');
const modeToggle = document.getElementById('modeToggle');
const modeText = document.getElementById('modeText');
const pumpRod = document.getElementById('pumpRod');
const pumpSpeed = document.getElementById('pumpSpeed');
const goodmanStatus = document.getElementById('goodmanStatus');

// Overlay Elements
const openOverlayBtn = document.getElementById('openOverlay');
const closeOverlayBtn = document.getElementById('closeOverlay');
const infoOverlay = document.getElementById('infoOverlay');

// Sidebar Elements
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebarToggle');

// Charts
let dynacardChart = null;
let fillageGauge = null;
let goodmanChart = null;
let ampsChart = null;
let featureChart = null;

// Latency Tracking
let lastPingTime = Date.now();

// Initialize Charts
function initCharts() {
    // 1. DUAL DYNACARD CHART (Line: Surface + Downhole)
    const ctxDynacard = document.getElementById('dynacardChart').getContext('2d');
    dynacardChart = new Chart(ctxDynacard, {
        type: 'line',
        data: {
            labels: Array(50).fill(''),
            datasets: [
                {
                    label: 'Surface Card',
                    data: [],
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true
                },
                {
                    label: 'Downhole Card',
                    data: [],
                    borderColor: '#ff0055',
                    backgroundColor: 'rgba(255, 0, 85, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    pointRadius: 0,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Position (%)', color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 9 } }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Load (kN)', color: '#64748b', font: { size: 10 } },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 9 } }
                }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 200 }
        }
    });

    // 2. FILLAGE GAUGE (Half Doughnut)
    const ctxFillage = document.getElementById('fillageGauge').getContext('2d');
    fillageGauge = new Chart(ctxFillage, {
        type: 'doughnut',
        data: {
            labels: ['Fillage', 'Empty'],
            datasets: [{
                data: [0, 100],
                backgroundColor: ['#00f2ff', 'rgba(255, 255, 255, 0.05)'],
                borderWidth: 0,
                circumference: 180,
                rotation: 270
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '80%',
            plugins: { legend: { display: false }, tooltip: { enabled: false } }
        }
    });

    // 3. GOODMAN DIAGRAM (Scatter with safe zone)
    const ctxGoodman = document.getElementById('goodmanChart').getContext('2d');
    goodmanChart = new Chart(ctxGoodman, {
        type: 'scatter',
        data: {
            datasets: [
                // Safe zone triangle (filled polygon approximation)
                {
                    label: 'Safe Zone',
                    data: [
                        { x: 0, y: 0 }, { x: 80, y: 0 }, { x: 40, y: 50 }, { x: 0, y: 50 }, { x: 0, y: 0 }
                    ],
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    borderColor: 'rgba(0, 255, 136, 0.3)',
                    borderWidth: 1,
                    showLine: true,
                    fill: true,
                    pointRadius: 0
                },
                // Current stress point
                {
                    label: 'Current Stress',
                    data: [{ x: 30, y: 60 }],
                    backgroundColor: '#00f2ff',
                    borderColor: '#00f2ff',
                    pointRadius: 8,
                    pointStyle: 'circle'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Min Stress (MPa)', color: '#64748b', font: { size: 9 } },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 8 } },
                    min: 0, max: 100
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Max Stress (MPa)', color: '#64748b', font: { size: 9 } },
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 8 } },
                    min: 0, max: 100
                }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 300 }
        }
    });

    // 4. AMPS CHART (Neon Line)
    const ctxAmps = document.getElementById('ampsChart').getContext('2d');
    ampsChart = new Chart(ctxAmps, {
        type: 'line',
        data: {
            labels: Array(50).fill(''),
            datasets: [{
                label: 'Amps',
                data: Array(50).fill(0),
                borderColor: '#00f2ff',
                backgroundColor: 'rgba(0, 242, 255, 0.1)',
                borderWidth: 1.5,
                tension: 0.4,
                pointRadius: 0,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 8 } }
                }
            },
            plugins: { legend: { display: false } },
            animation: { duration: 0 }
        }
    });

    // 5. FEATURE IMPORTANCE CHART (Horizontal Bar)
    const ctxFeature = document.getElementById('featureChart').getContext('2d');
    featureChart = new Chart(ctxFeature, {
        type: 'bar',
        data: {
            labels: ['Dynacard Shape', 'Fillage %', 'Vibration', 'Motor Amps', 'Rod Load'],
            datasets: [{
                label: 'SHAP Value',
                data: [0.85, 0.72, 0.45, 0.30, 0.25],
                backgroundColor: [
                    '#00f2ff', '#00ff88', '#ff8800', '#ff0055', '#ffd700'
                ],
                borderWidth: 0,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    grid: { color: 'rgba(255, 255, 255, 0.03)' },
                    ticks: { color: '#64748b', font: { size: 8 } }
                },
                y: {
                    display: true,
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 9 } }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// WebSocket Connection
function connect() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/websocket`);

    ws.onopen = () => {
        connectionStatus.classList.add('online');
        systemStatus.textContent = 'ONLINE';
        systemStatus.style.color = '#00f2ff';
        console.log('[DIGITAL TWIN] Connected to Mission Control');
    };

    ws.onmessage = (event) => {
        const now = Date.now();
        latencyEl.textContent = (now - lastPingTime);
        lastPingTime = now;

        const data = JSON.parse(event.data);
        updateDashboard(data);
    };

    ws.onclose = () => {
        connectionStatus.classList.remove('online');
        systemStatus.textContent = 'OFFLINE';
        systemStatus.style.color = '#ff0055';
        setTimeout(connect, 3000);
    };

    ws.onerror = (error) => {
        console.error('[DIGITAL TWIN] Connection Error:', error);
    };
}

// Update Dashboard UI
function updateDashboard(data) {
    const analysis = data.analysis;

    // 1. Timestamp
    lastUpdated.textContent = data.timestamp;

    // 2. DUAL DYNACARD (Zone 1: The Pulse)
    if (analysis.dynacard_physics) {
        const surface = analysis.dynacard_physics.surface;
        const downhole = analysis.dynacard_physics.downhole;

        // Update chart labels and data
        dynacardChart.data.labels = surface.map(pt => pt[0].toFixed(0));
        dynacardChart.data.datasets[0].data = surface.map(pt => pt[1]);
        dynacardChart.data.datasets[1].data = downhole.map(pt => pt[1]);
        dynacardChart.update();

        // Pump Animation (sync with stroke position)
        const strokePos = analysis.dynacard_physics.stroke_position;
        pumpRod.style.left = `${strokePos}%`;
    }

    // 3. AI Diagnosis (Zone 3: The Hands)
    if (analysis.classification) {
        const diag = analysis.classification.diagnosis;
        aiDiagnosis.textContent = diag;
        const confPercent = Math.round(analysis.classification.confidence * 100);
        confidenceFill.style.width = `${confPercent}%`;

        if (diag.includes("Normal")) {
            aiDiagnosis.classList.remove('alert');
            aiDiagnosis.style.color = "#00ff88";
        } else {
            aiDiagnosis.classList.add('alert');
            aiDiagnosis.style.color = "#ff0055";
        }
    }

    // 4. XAI Reason (Explainable AI)
    if (analysis.xai_reason) {
        xaiReason.innerHTML = analysis.xai_reason.split(' | ').map(r => `• ${r}`).join('<br>');
    }

    // 5. Fillage Gauge (Zone 2: The Brain)
    if (analysis.control) {
        const fillage = analysis.control.fillage;
        fillageValue.textContent = `${fillage.toFixed(0)}%`;

        fillageGauge.data.datasets[0].data = [fillage, 100 - fillage];
        if (fillage < 70) {
            fillageGauge.data.datasets[0].backgroundColor = ['#ff0055', 'rgba(255, 255, 255, 0.05)'];
            fillageValue.style.color = '#ff0055';
        } else if (fillage < 85) {
            fillageGauge.data.datasets[0].backgroundColor = ['#ff8800', 'rgba(255, 255, 255, 0.05)'];
            fillageValue.style.color = '#ff8800';
        } else {
            fillageGauge.data.datasets[0].backgroundColor = ['#00f2ff', 'rgba(255, 255, 255, 0.05)'];
            fillageValue.style.color = '#00f2ff';
        }
        fillageGauge.update();

        // Pump Speed
        pumpSpeed.textContent = `${analysis.control.current_spm.toFixed(1)} SPM`;

        // Action Log
        if (analysis.control.action !== "Monitoring") {
            addLogEntry(data.timestamp, analysis.control.description, 'action');
        }
    }

    // 6. Goodman Diagram (Zone 2: Structural Safety)
    if (analysis.goodman) {
        const { min_stress, max_stress, is_safe } = analysis.goodman;

        // Update stress point
        goodmanChart.data.datasets[1].data = [{ x: min_stress, y: max_stress }];
        goodmanChart.data.datasets[1].backgroundColor = is_safe ? '#00f2ff' : '#ff0055';
        goodmanChart.data.datasets[1].borderColor = is_safe ? '#00f2ff' : '#ff0055';
        goodmanChart.update();

        // Update status indicator
        const statusDot = goodmanStatus.querySelector('.status-dot');
        const statusText = goodmanStatus.querySelector('.status-text');
        if (is_safe) {
            statusDot.className = 'status-dot safe';
            statusText.textContent = 'SAFE';
            statusText.style.color = '#00ff88';
        } else {
            statusDot.className = 'status-dot danger';
            statusText.textContent = 'DANGER';
            statusText.style.color = '#ff0055';
        }
    }

    // 7. Signal Confidence (SNR indicator)
    if (analysis.signal_confidence) {
        const { level, snr_db, bars } = analysis.signal_confidence;
        signalIndicator.className = `signal-indicator ${level}`;
        snrValue.textContent = snr_db.toFixed(1);
    }

    // 8. Motor Amps (Zone 3)
    if (analysis.amps) {
        ampsChart.data.datasets[0].data = analysis.amps;
        ampsChart.update();
    }

    // 9. Financial Metrics
    if (analysis.financial) {
        strokesSaved.textContent = analysis.financial.strokes_saved;
        energySaved.textContent = `${analysis.financial.energy_saved.toFixed(2)} kWh`;
    }
}

// Add Log Entry
let lastLogMsg = "";
function addLogEntry(time, message, type) {
    if (message === lastLogMsg) return;
    lastLogMsg = message;

    const parts = message.split(' -> ');
    const event = parts[0] || message;
    const action = parts[1] || '';

    const entry = document.createElement('div');
    entry.className = `log-item ${type}`;
    entry.innerHTML = `
        <span class="log-time mono">[${time}]</span>
        <span class="log-event">${event}</span>
        ${action ? `<span class="log-action">→ ${action}</span>` : ''}
    `;

    actionLog.prepend(entry);

    if (actionLog.children.length > 30) {
        actionLog.lastElementChild.remove();
    }
}

// Mode Toggle Handler
modeToggle.addEventListener('change', async () => {
    const isAutonomous = modeToggle.checked;
    const mode = isAutonomous ? 'autonomous' : 'advisory';

    modeText.textContent = mode.toUpperCase();
    modeText.style.color = isAutonomous ? '#00f2ff' : '#ff8800';

    // Call API to set mode
    try {
        await fetch(`/api/set_mode?mode=${mode}`, { method: 'POST' });
        addLogEntry(new Date().toLocaleTimeString(), `Mode changed to ${mode.toUpperCase()}`, isAutonomous ? 'action' : 'warning');
    } catch (e) {
        console.error('[DIGITAL TWIN] Mode toggle error:', e);
    }
});

// Overlay Toggle Handlers
openOverlayBtn.addEventListener('click', () => {
    infoOverlay.classList.add('active');
});

closeOverlayBtn.addEventListener('click', () => {
    infoOverlay.classList.remove('active');
});

// Close on backdrop click
infoOverlay.addEventListener('click', (e) => {
    if (e.target === infoOverlay) {
        infoOverlay.classList.remove('active');
    }
});

// Sidebar Toggle
if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
        sidebar.classList.toggle('active');
        sidebarToggle.textContent = sidebar.classList.contains('active') ? '×' : '☰';
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connect();
});
