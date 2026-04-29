"""
ML Model Loader untuk SmartFarm API — TensorFlow/Keras LSTM Edition
===================================================================
Memuat model LSTM (classification & forecasting) yang di-train
menggunakan notebook Classification_V3_Final.ipynb dan Forecasting_Final.ipynb.

Classification: LSTM Balanced (Dampened) — 21 fitur, window=12
Forecasting:    LSTM Balanced — 11 fitur, window=6, resampled 30min
"""
import logging
import joblib
import numpy as np
from pathlib import Path
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "trained_models"

# ─── Global state ─────────────────────────────────────────────────────────────
classification_model = None
classification_scaler = None
forecasting_model = None
forecasting_scaler = None

# Sliding-window buffers (per-kandang bisa diperluas di masa depan)
_cls_raw_buffer: deque = deque(maxlen=20)   # raw features sebelum FE
_fc_raw_buffer: deque = deque(maxlen=20)    # raw sensor for forecasting

# ─── Configuration (harus sama persis dengan notebook) ────────────────────────
CLS_TIME_STEPS = 12
CLS_ROLLING_WINDOW = 6
CLS_SMOOTH_WINDOW = 5

FC_TIME_STEPS = 6
FC_ROLLING_WINDOW = 3

# Classification feature order — HARUS sesuai urutan saat training
CLS_FEATURES = [
    'Suhu', 'Kelembaban', 'Amoniak', 'Pakan', 'Minum', 'Bobot',
    'Populasi', 'Luas Kandang',
    'Hour', 'Session', 'Feed_Water_Ratio', 'Density',
    'delta_Suhu', 'delta_Kelembaban', 'delta_Amoniak',
    'rolling_mean_Suhu', 'rolling_mean_Kelembaban', 'rolling_mean_Amoniak',
    'rolling_std_Suhu', 'rolling_std_Kelembaban', 'rolling_std_Amoniak',
]

# Forecasting feature order — HARUS sesuai urutan saat training
FC_FEATURES = [
    'Suhu', 'Kelembaban', 'Amoniak',
    'Hour',
    'delta_Suhu', 'delta_Kelembaban', 'delta_Amoniak',
    'rolling_mean_Suhu', 'rolling_mean_Kelembaban', 'rolling_mean_Amoniak',
    'Death',
]


# ─── Load Models ──────────────────────────────────────────────────────────────
def load_models():
    """Load semua trained models (dipanggil saat startup atau reload)."""
    global classification_model, classification_scaler
    global forecasting_model, forecasting_scaler

    try:
        import tensorflow as tf
        # Suppress TF warnings
        tf.get_logger().setLevel('ERROR')

        cls_path = MODEL_DIR / "best_model_classification.h5"
        fc_path = MODEL_DIR / "best_model_death_forecasting.h5"
        cls_scaler_path = MODEL_DIR / "scaler_classification.pkl"
        fc_scaler_path = MODEL_DIR / "scaler_forecasting.pkl"

        if cls_path.exists():
            classification_model = tf.keras.models.load_model(str(cls_path), compile=False)
            logger.info(f"✅ Classification model loaded: {cls_path.name}")
        else:
            logger.warning(f"⚠️  Classification model not found: {cls_path}")

        if fc_path.exists():
            forecasting_model = tf.keras.models.load_model(str(fc_path), compile=False)
            logger.info(f"✅ Forecasting model loaded: {fc_path.name}")
        else:
            logger.warning(f"⚠️  Forecasting model not found: {fc_path}")

        if cls_scaler_path.exists():
            classification_scaler = joblib.load(cls_scaler_path)
            logger.info(f"✅ Classification scaler loaded: {cls_scaler_path.name}")
        else:
            logger.warning(f"⚠️  Classification scaler not found: {cls_scaler_path}")

        if fc_scaler_path.exists():
            forecasting_scaler = joblib.load(fc_scaler_path)
            logger.info(f"✅ Forecasting scaler loaded: {fc_scaler_path.name}")
        else:
            logger.warning(f"⚠️  Forecasting scaler not found: {fc_scaler_path}")

        logger.info("✅ ML Models loading complete")
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        logger.error("Pastikan file .h5 dan .pkl ada di folder trained_models/")


# ─── Feature Engineering Helpers ──────────────────────────────────────────────
def _hour_to_session(hour: int) -> int:
    """Convert jam ke session: 0=malam(0-6), 1=pagi(6-12), 2=siang(12-18), 3=sore(18-24)."""
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3


def _compute_classification_features(raw_points: list) -> Optional[np.ndarray]:
    if len(raw_points) < CLS_TIME_STEPS + CLS_ROLLING_WINDOW:
        return None

    import pandas as pd
    df = pd.DataFrame(raw_points)

    df['Session'] = df['Hour'].apply(_hour_to_session)
    df['Feed_Water_Ratio'] = df['Pakan'] / (df['Minum'] + 1)
    df['Density'] = df['Populasi'] / (df['Luas Kandang'] + 1)

    for col in ['Suhu', 'Kelembaban', 'Amoniak']:
        df[f'delta_{col}'] = df[col].diff().fillna(0)

    for col in ['Suhu', 'Kelembaban', 'Amoniak']:
        df[f'rolling_mean_{col}'] = df[col].rolling(window=CLS_ROLLING_WINDOW, min_periods=1).mean()
        df[f'rolling_std_{col}'] = df[col].rolling(window=CLS_ROLLING_WINDOW, min_periods=1).std().fillna(0)

    feature_matrix = df[CLS_FEATURES].values
    feature_matrix_scaled = classification_scaler.transform(feature_matrix)

    window = feature_matrix_scaled[-CLS_TIME_STEPS:]
    return window.reshape(1, CLS_TIME_STEPS, len(CLS_FEATURES))


def _compute_forecasting_features(raw_points: list) -> Optional[np.ndarray]:
    if len(raw_points) < FC_TIME_STEPS + FC_ROLLING_WINDOW:
        return None

    import pandas as pd
    df = pd.DataFrame(raw_points)

    for col in ['Suhu', 'Kelembaban', 'Amoniak']:
        df[f'delta_{col}'] = df[col].diff().fillna(0)

    for col in ['Suhu', 'Kelembaban', 'Amoniak']:
        df[f'rolling_mean_{col}'] = df[col].rolling(window=FC_ROLLING_WINDOW, min_periods=1).mean()

    feature_matrix = df[FC_FEATURES].values
    feature_matrix_scaled = forecasting_scaler.transform(feature_matrix)

    window = feature_matrix_scaled[-FC_TIME_STEPS:]
    return window.reshape(1, FC_TIME_STEPS, len(FC_FEATURES))


# ─── Prediction Functions (API tetap sama!) ───────────────────────────────────
def predict_classification(features: dict) -> dict:
    if classification_model is None or classification_scaler is None:
        load_models()
    
    if classification_model is None:
        raise RuntimeError("Classification model belum ter-load. Cek file .h5")

    raw_point = {
        'Suhu': float(features.get('Suhu', 0)),
        'Kelembaban': float(features.get('Kelembaban', 0)),
        'Amoniak': float(features.get('Amoniak', 0)),
        'Pakan': float(features.get('Pakan', 0)),
        'Minum': float(features.get('Minum', 0)),
        'Bobot': float(features.get('Bobot', 0)),
        'Populasi': float(features.get('Populasi', 0)),
        'Luas Kandang': float(features.get('Luas Kandang', 120)),
        'Hour': int(features.get('Hour', 12)),
    }
    _cls_raw_buffer.append(raw_point)

    X = _compute_classification_features(list(_cls_raw_buffer))

    if X is None:
        needed = CLS_TIME_STEPS + CLS_ROLLING_WINDOW
        padded = [raw_point] * needed
        X = _compute_classification_features(padded)
        if X is None:
            logger.warning("Classification: tidak bisa membangun window, fallback Normal")
            return {'class': 'Normal', 'probability': 0.5, 'confidence': 0.5}

    pred_prob = float(classification_model.predict(X, verbose=0)[0][0])
    pred_class = 'Abnormal' if pred_prob > 0.5 else 'Normal'

    return {
        'class': pred_class,
        'probability': pred_prob if pred_class == 'Abnormal' else (1 - pred_prob),
        'confidence': max(pred_prob, 1 - pred_prob),
    }


def predict_forecasting(sensor_history: list) -> dict:
    if forecasting_model is None or forecasting_scaler is None:
        load_models()

    if forecasting_model is None:
        raise RuntimeError("Forecasting model belum ter-load. Cek file .h5")

    from datetime import datetime
    current_hour = datetime.now().hour

    for point in sensor_history:
        raw_point = {
            'Suhu': float(point.get('temp', 0)),
            'Kelembaban': float(point.get('hum', 0)),
            'Amoniak': float(point.get('ammo', 0)),
            'Hour': current_hour,
            'Death': float(point.get('Death', 0)),
        }
        _fc_raw_buffer.append(raw_point)

    X = _compute_forecasting_features(list(_fc_raw_buffer))

    if X is None:
        needed = FC_TIME_STEPS + FC_ROLLING_WINDOW
        last_point = {
            'Suhu': float(sensor_history[-1].get('temp', 0)),
            'Kelembaban': float(sensor_history[-1].get('hum', 0)),
            'Amoniak': float(sensor_history[-1].get('ammo', 0)),
            'Hour': current_hour,
            'Death': float(sensor_history[-1].get('Death', 0)),
        }
        padded = [last_point] * needed
        X = _compute_forecasting_features(padded)
        if X is None:
            logger.warning("Forecasting: tidak bisa membangun window, fallback 0")
            return {'predicted_death': 0, 'raw_prediction': 0.0}

    pred_scaled = float(forecasting_model.predict(X, verbose=0)[0][0])

    death_idx = FC_FEATURES.index('Death')
    dummy = np.zeros((1, len(FC_FEATURES)))
    dummy[0, death_idx] = pred_scaled
    pred_original = forecasting_scaler.inverse_transform(dummy)[0, death_idx]

    final_pred = int(round(max(0, pred_original)))

    return {
        'predicted_death': final_pred,
        'raw_prediction': float(pred_original),
    }
