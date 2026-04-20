"""
SmartFarm ML Prediction Endpoints
Integrated with trained RandomForest (Classification) and XGBoost (Forecasting) models
"""
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.base import BaseResponse, success_response
from app.services.activity_log_service import ActivityLogService
from app.services.notification_service import NotificationService
from app.services.prediction_service import PredictionService
from app.models.kandang import Kandang
from app.api.deps import get_current_user, get_request_info, get_single_kandang
from app.ml.model_loader import predict_classification, predict_forecasting, load_models
import uuid

router = APIRouter()


# ============================================
# REQUEST SCHEMAS
# ============================================

class ClassificationRequest(BaseModel):
    hari_ke: int = Field(..., ge=1)
    suhu: float = Field(..., ge=0, le=60)
    kelembaban: float = Field(..., ge=0, le=100)
    amoniak: float = Field(..., ge=0)
    pakan: float = Field(..., ge=0)
    minum: float = Field(..., ge=0)
    bobot: float = Field(..., ge=0)
    populasi: int = Field(..., ge=1)
    luas_kandang: float = Field(..., gt=0)
    hour: int = Field(..., ge=0, le=23)

    class Config:
        json_schema_extra = {
            "example": {
                "hari_ke": 5,
                "suhu": 28.5,
                "kelembaban": 75.0,
                "amoniak": 3.5,
                "pakan": 150,
                "minum": 350,
                "bobot": 58,
                "populasi": 8000,
                "luas_kandang": 336,
                "hour": 10,
            }
        }


class SensorDataPoint(BaseModel):
    """Single sensor data point for forecasting history."""
    temp: float = Field(..., description="Temperature (°C)")
    hum: float = Field(..., description="Humidity (%)")
    ammo: float = Field(..., description="Ammonia level (ppm)")
    Death: int = Field(..., ge=0, description="Number of deaths in this interval")


class ForecastingRequest(BaseModel):
    sensor_history: List[SensorDataPoint] = Field(..., min_length=4, max_length=10)

    class Config:
        json_schema_extra = {
            "example": {
                "sensor_history": [
                    {"temp": 27.5, "hum": 72.0, "ammo": 3.2, "Death": 0},
                    {"temp": 27.8, "hum": 71.5, "ammo": 3.4, "Death": 0},
                    {"temp": 28.0, "hum": 70.0, "ammo": 3.6, "Death": 1},
                    {"temp": 28.2, "hum": 69.5, "ammo": 3.8, "Death": 0}
                ]
            }
        }


# ============================================
# RESPONSE SCHEMAS
# ============================================

class ClassificationResponse(BaseModel):
    """Response schema for classification prediction."""
    prediction: str = Field(..., description="Predicted class: 'Normal' or 'Abnormal'")
    probability: float = Field(..., ge=0, le=1, description="Probability of predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (max probability)")
    is_abnormal: bool = Field(..., description="True if condition is abnormal")


class ForecastingResponse(BaseModel):
    """Response schema for forecasting prediction."""
    predicted_death: int = Field(..., ge=0, description="Predicted number of deaths")
    raw_prediction: float = Field(..., description="Raw model output (unrounded)")
    has_risk: bool = Field(..., description="True if predicted_death > 0")


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""
    classification_model: Dict[str, Any] = Field(..., description="Classification model info")
    forecasting_model: Dict[str, Any] = Field(..., description="Forecasting model info")


# ============================================
# ENDPOINTS
# ============================================

@router.post(
    "/classify",
    response_model=BaseResponse[ClassificationResponse],
    summary="Classify Condition",
    description="Classify farm condition (Normal/Abnormal) based on sensor data using RandomForest model",
)
async def classify_condition(
    request: Request,
    data: ClassificationRequest,
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    """
    Classify the current farm condition based on sensor readings.
    
    Uses a trained RandomForest classifier with StandardScaler preprocessing.
    Returns 'Normal' or 'Abnormal' classification with confidence scores.
    """
    try:
        # Prepare features for model
        features = {
            'Hari Ke-': data.hari_ke,
            'Suhu': data.suhu,
            'Kelembaban': data.kelembaban,
            'Amoniak': data.amoniak,
            'Pakan': data.pakan,
            'Minum': data.minum,
            'Bobot': data.bobot,
            'Populasi': data.populasi,
            'Luas Kandang': data.luas_kandang,
            'Hour': data.hour
        }
        
        # Get prediction from model
        result = predict_classification(features)
        
        prediction_result = ClassificationResponse(
            prediction=result['class'],
            probability=result['probability'],
            confidence=result['confidence'],
            is_abnormal=(result['class'] == 'Abnormal')
        )
        
        # Log activity
        request_info = await get_request_info(request)
        activity_service = ActivityLogService(db)
        await activity_service.log_action(
            user_id=current_user.id,
            action="classify",
            resource="prediction",
            request_info=request_info,
            details={
                "input": data.model_dump(),
                "prediction": prediction_result.prediction,
                "confidence": prediction_result.confidence,
            },
        )
        
        if prediction_result.is_abnormal:
            notification_service = NotificationService(db)
            await notification_service.create_classification_alert(
                user_id=current_user.id,
                kandang_id=kandang.id,
                prediction=prediction_result.prediction,
                sensor_data=features,
            )
        
        return success_response(
            data=prediction_result,
            message=f"Klasifikasi berhasil: {prediction_result.prediction}",
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification error: {str(e)}"
        )


@router.post(
    "/forecast",
    response_model=BaseResponse[ForecastingResponse],
    summary="Forecast Deaths",
    description="Forecast number of chicken deaths based on sensor history using XGBoost model",
)
async def forecast_mortality(
    request: Request,
    data: ForecastingRequest,
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    """
    Forecast mortality based on 2 hours of sensor history.
    
    Uses a trained XGBoost regressor with statistical feature engineering.
    Requires 4 data points (30-minute intervals over 2 hours).
    """
    try:
        # Convert to format expected by model
        sensor_history = [
            {
                'temp': point.temp,
                'hum': point.hum,
                'ammo': point.ammo,
                'Death': point.Death
            }
            for point in data.sensor_history
        ]
        
        # Get prediction from model
        result = predict_forecasting(sensor_history)
        
        forecast_result = ForecastingResponse(
            predicted_death=result['predicted_death'],
            raw_prediction=result['raw_prediction'],
            has_risk=(result['predicted_death'] > 0)
        )
        
        # Log activity
        request_info = await get_request_info(request)
        activity_service = ActivityLogService(db)
        await activity_service.log_action(
            user_id=current_user.id,
            action="forecast",
            resource="prediction",
            request_info=request_info,
            details={
                "input_count": len(data.sensor_history),
                "predicted_death": forecast_result.predicted_death,
            },
        )
        
        if forecast_result.has_risk:
            notification_service = NotificationService(db)
            await notification_service.create_death_forecast_alert(
                user_id=current_user.id,
                kandang_id=kandang.id,
                predicted_death=forecast_result.predicted_death,
                raw_prediction=forecast_result.raw_prediction,
            )
        
        return success_response(
            data=forecast_result,
            message=f"Forecasting berhasil: {forecast_result.predicted_death} kematian diprediksi",
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting error: {str(e)}"
        )


@router.get(
    "/models",
    response_model=BaseResponse[ModelInfoResponse],
    summary="Get Model Info",
    description="Get information about loaded ML models",
)
async def get_models_info(
    current_user: User = Depends(get_current_user),
):
    """
    Get information about available ML models.
    """
    model_info = ModelInfoResponse(
        classification_model={
            "name": "SmartFarm Classification Model",
            "type": "RandomForestClassifier",
            "version": "1.0.0",
            "status": "loaded",
            "description": "Classification model for farm condition (Normal/Abnormal)",
            "input_features": [
                "Hari Ke-", "Suhu", "Kelembaban", "Amoniak", "Pakan", 
                "Minum", "Bobot", "Populasi", "Luas Kandang", "Hour"
            ],
            "output_classes": ["Normal", "Abnormal"],
            "metrics": {
                "accuracy": "97.21%",
                "f1_score": "93.02%"
            }
        },
        forecasting_model={
            "name": "SmartFarm Forecasting Model",
            "type": "XGBRegressor",
            "version": "1.0.0",
            "status": "loaded",
            "description": "Forecasting model for chicken death prediction",
            "input_features": ["temp", "hum", "ammo", "Death (history)"],
            "window_size": "4 data points (2 hours)",
            "output": "predicted_death (count)",
            "metrics": {
                "rmse": "0.2752",
                "pearson": "0.7351"
            }
        },
    )
    
    return success_response(
        data=model_info,
        message="Informasi model berhasil diambil",
    )


@router.post(
    "/load-models",
    summary="Load/Reload ML Models",
    description="Manually load or reload ML models (Admin only)",
)
async def reload_models(
    current_user: User = Depends(get_current_user),
):
    """
    Reload ML models from disk.
    Useful after updating model files.
    """
    # Check admin role
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin can reload models"
        )
    
    try:
        load_models()
        return success_response(
            data={"status": "models_reloaded"},
            message="ML models berhasil di-reload",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload models: {str(e)}"
        )


@router.get(
    "/history",
    summary="Get Prediction History",
    description="Riwayat hasil prediksi ML dari IoT (classification & forecasting)",
)
async def get_prediction_history(
    type: Optional[str] = None,
    limit: int = 1000,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    _: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    import json
    import datetime as dt

    start = start_date.date() if start_date else None
    end = end_date.date() if end_date else None

    svc = PredictionService(db)
    records = await svc.get_history(kandang.id, limit=limit, prediction_type=type, start_date=start, end_date=end)

    data = []
    for r in records:
        data.append({
            "id": str(r.id),
            "type": r.type,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "predicted_death": r.predicted_death,
            "raw_prediction": r.raw_prediction,
            "input_data": json.loads(r.input_data) if r.input_data else None,
            "created_at": r.created_at.isoformat(),
        })

    return success_response(data=data, message=f"{len(data)} riwayat prediksi")
