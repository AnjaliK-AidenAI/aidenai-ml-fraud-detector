"""
AidenAI ML Fraud Detector â€” FastAPI service
Exposes /predict for real-time fraud scoring and /train to retrain the model.
"""
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from app.model import FraudModel
from app.schemas import TransactionIn, PredictionOut, TrainResponse, HealthOut

MODEL_PATH = Path("models/fraud_model.joblib")
_model = FraudModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        _model.load(MODEL_PATH)
        print(f"[startup] model loaded from {MODEL_PATH}")
    else:
        print("[startup] no saved model found â€” call POST /train first")
    yield


app = FastAPI(
    title="AidenAI Fraud Detector",
    version="1.0.0",
    description="Real-time transaction fraud detection powered by scikit-learn",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthOut)
def health():
    return HealthOut(status="healthy", model_loaded=_model.is_trained)


@app.post("/predict", response_model=PredictionOut)
def predict(tx: TransactionIn):
    if not _model.is_trained:
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")
    prob, label = _model.predict(tx)
    return PredictionOut(
        transaction_id=tx.transaction_id,
        fraud_probability=round(prob, 4),
        is_fraud=label,
        risk_level="HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW",
    )


@app.post("/train", response_model=TrainResponse)
def train(background: BackgroundTasks):
    def _run():
        metrics = _model.train_on_synthetic()
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        _model.save(MODEL_PATH)
        print(f"[train] done  auc={metrics['roc_auc']:.3f}")

    background.add_task(_run)
    return TrainResponse(status="training_started", message="Training in background. Poll /health for model_loaded=true.")