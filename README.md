# Stellar Classification AI Web Application

A production-ready, AI-powered web application for classifying stellar objects (GALAXY, QSO, STAR) using a trained Random Forest machine learning model on SDSS astronomical data.

![Tech Stack](https://img.shields.io/badge/Next.js-14-black?style=flat-square&logo=next.js)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)
![Accuracy](https://img.shields.io/badge/Accuracy-97.3%25-cyan?style=flat-square)

## Overview

This platform allows users to enter astronomical feature values and receive accurate star classification predictions in real time. The UI features a futuristic space-themed design with glassmorphism, particle effects, and Framer Motion animations.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 14, TypeScript, Tailwind CSS, Framer Motion, Recharts |
| Backend | FastAPI, Pydantic, Uvicorn |
| ML | scikit-learn, imbalanced-learn, joblib |
| Deployment | Vercel (frontend), Render/Railway (backend) |

## Project Structure

```
Steller Classification/
├── frontend/          # Next.js web application
├── backend/           # FastAPI REST API
│   ├── app/           # Application code
│   └── scripts/       # Model training script
├── model/             # Trained model artifacts (.pkl)
├── dataset/           # SDSS star classification CSV
├── assets/            # Static assets
├── public/            # Public files
├── requirements.txt   # Python dependencies
├── package.json       # Root npm scripts
└── README.md
```

## Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.10+
- pip

## Installation

### 1. Clone and navigate to the project

```bash
cd "Steller Classification"
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML model (if not already present)

```bash
python backend/scripts/train_model.py
```

This creates `model/stellar_classifier.pkl`, `model/scaler.pkl`, and `model/model_metadata.pkl`.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cp .env.example .env.local
cd ..
```

## Running Locally

### Start the backend (Terminal 1)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs available at: http://localhost:8000/docs

### Start the frontend (Terminal 2)

```bash
cd frontend
npm run dev
```

Open http://localhost:3000 in your browser.

### Or use root scripts

```bash
npm run dev:backend    # Terminal 1
npm run dev:frontend   # Terminal 2
```

## API Documentation

### `POST /predict`

Classify a stellar object based on astronomical features.

**Request Body:**

```json
{
  "alpha": 39.15,
  "delta": 28.1,
  "u": 21.75,
  "g": 20.03,
  "r": 19.18,
  "i": 18.82,
  "z": 18.65,
  "redshift": 0.0
}
```

**Response:**

```json
{
  "prediction": "STAR",
  "confidence": 0.9842,
  "probabilities": [
    { "label": "STAR", "probability": 0.9842 },
    { "label": "GALAXY", "probability": 0.0123 },
    { "label": "QSO", "probability": 0.0035 }
  ]
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and model status |
| GET | `/model-info` | Model metadata (features, classes, accuracy) |

## Environment Variables

### Backend (`backend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated allowed origins |
| `LOG_LEVEL` | `INFO` | Logging level |

### Frontend (`frontend/.env.local`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API URL |

## Deployment

### Frontend — Vercel

1. Push the project to GitHub
2. Import the `frontend/` directory in Vercel
3. Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend-url.com`
4. Deploy

### Backend — Render

1. Create a new **Web Service** on Render
2. Set root directory to `backend`
3. Build command: `pip install -r ../requirements.txt && python ../backend/scripts/train_model.py`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables:
   - `CORS_ORIGINS=https://your-frontend.vercel.app`

### Backend — Railway

1. Connect your GitHub repo
2. Set the service root to `backend`
3. Use the included `Procfile`
4. Configure `CORS_ORIGINS` with your frontend URL

## ML Model Details

- **Algorithm:** Random Forest Classifier (100 estimators, max depth 20)
- **Preprocessing:** StandardScaler + SMOTE-Tomek (training only)
- **Features:** alpha, delta, u, g, r, i, z, redshift (8 features)
- **Classes:** GALAXY, QSO, STAR
- **Test Accuracy:** 97.31%
- **Dataset:** SDSS star classification (100,000 samples)

## Pages

| Route | Description |
|-------|-------------|
| `/` | Landing page with hero, features, how-it-works, accuracy, about |
| `/dashboard` | Prediction interface with inputs, validation, and results |

## License

This project is for educational and research purposes.
