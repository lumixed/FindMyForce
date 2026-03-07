# Find My Force — RF COP System

RF Signal Classification & Geolocation — UBC Defence Tech Hackathon 2026

## Quick Start

```bash
# 1. Set up virtual environment (already done if you ran setup)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set your API key
echo "API_KEY=your-team-api-key" >> .env

# 3. Place training data in data/ directory
#    Download the HDF5 file and put it at data/training.h5

# 4. Train the classifier
python3 main.py train

# 5. Start the dashboard
python3 main.py server --port 5050
# → Open http://localhost:5050
```

## Architecture

```
LockedIn_FindMyForce/
├── classifier/
│   ├── signal_classifier.py    # ML pipeline: feature extraction + RF classifier + OCC anomaly detector
│   └── __init__.py
├── pipeline/
│   ├── geolocator.py           # RSSI trilateration, TDoA multilateration, Kalman filter
│   ├── track_manager.py        # Persistent emitter track lifecycle management
│   ├── associator.py           # Multi-receiver observation grouping
│   ├── feed_consumer.py        # SSE stream consumer + submission manager
│   └── __init__.py
├── dashboard/
│   ├── index.html              # COP dashboard
│   ├── style.css               # Tactical dark UI
│   └── app.js                  # Leaflet + Socket.IO frontend
├── data/                       # Place training HDF5 here
├── models/                     # Saved classifier (auto-created after training)
├── server.py                   # Flask + Socket.IO backend
├── main.py                     # CLI entry point
├── requirements.txt
└── .env                        # API_KEY and API_URL
```

## CLI Commands

```bash
# Start dashboard server
python3 main.py server [--port 5050] [--debug]

# Train classifier on HDF5 data
python3 main.py train

# Stream live feed observations (debug)
python3 main.py stream

# Fetch team score
python3 main.py score
```

## ML Pipeline

### Signal Classifier
- **Feature extraction** (42 features): amplitude envelope stats, phase/frequency features, spectral analysis (FFT), pulsed signal detection, duty cycle, BPSK/FMCW/ASK pattern detection, IQ correlation
- **Random Forest** (300 trees): classifies known friendly signal types
- **One-Class SVM**: trained only on friendly data — detects out-of-distribution (hostile/civilian) signals as anomalies

### Training Procedure
1. Load HDF5 data → extract 42-dim feature vectors
2. Train RF classifier on friendly labels
3. Train OC-SVM on friendly features only
4. Calibrate anomaly threshold at 10th percentile of friendly OOD scores
5. Save both models to `models/classifier.joblib`

## Geolocation Engine

### RSSI Trilateration
- Converts RSSI to distance via path-loss model: `d = d_ref × 10^((RSSI_ref - RSSI) / (10 × n))`
- Nonlinear least-squares optimization (scipy `least_squares`)
- SNR-weighted receiver contributions
- Fallback to 2-receiver and single-receiver modes

### TDoA Multilateration
- Uses time-of-arrival differences to compute hyperbolic position fixes
- Linearized initial estimate, then nonlinear refinement
- Requires ≥3 receivers with valid ToA

### Hybrid Fusion
- 70% TDoA + 30% RSSI weighted position average when both methods available
- GDOP (Geometric Dilution of Precision) estimation

### Kalman Filter Tracking
- State: [x, y, vx, vy] (position + velocity)
- Constant velocity motion model
- Measurement update weighted by geolocation uncertainty

## Track Management

- **TENTATIVE** → **CONFIRMED** after 2 updates
- **CONFIRMED** → **COASTING** after 20s without updates
- **COASTING** → **LOST** after 60s
- Kalman-smoothed position estimates
- Classification confidence EMA fusion across observations

## Observation Association

Groups independent receiver observations into single-emitter groups using:
1. **Temporal gate**: observations within 500ms window
2. **Classification gate**: same signal type (or at least one "unknown")
3. **IQ similarity gate**: cosine similarity > 0.5

## Scoring Strategy

- **Classification (40%)**: Correctly identifies friendly types AND detects hostile/civilian as anomalies
- **Geolocation (30%)**: RSSI + TDoA hybrid with Kalman smoothing minimizes CEP
- **Novelty (30%)**: OC-SVM flags unknowns; use intelligence briefing labels for hostile sub-types

### Hostile Signal Labels (for eval submission)
- `Airborne-detection` — Airborne surveillance radar
- `Airborne-range` — Airborne range-finding radar
- `Air-Ground-MTI` — Air-to-ground moving target indicator
- `EW-Jammer` — Electronic warfare / broadband jammer
- `AM radio` — Commercial AM radio broadcast

## Dashboard Features

- **Leaflet map** with dark tactical overlay
- Real-time track markers color-coded by affiliation (green=friendly, red=hostile, orange=unknown, blue=civilian)
- Uncertainty radius circles around each track
- Track history path polylines
- Receiver station markers with tooltip details
- Live observation feed panel
- Track detail panel with classification confidence, GDOP, velocity
- Track filter by affiliation
- Score display with per-component breakdown
- One-click eval submission button
- Socket.IO real-time updates (<100ms latency)
