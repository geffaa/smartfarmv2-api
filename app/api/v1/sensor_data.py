"""
Sensor Data API Routes - IoT sensor data CRUD endpoints
"""
import uuid
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.base import BaseResponse, success_response
from app.schemas.sensor_data import (
    SensorDataCreate,
    SensorDataIoTCreate,
    SensorDataManualUpdate,
    SensorDataResponse,
    SensorDataListResponse,
    SensorDataStats,
)
from app.services.sensor_data_service import SensorDataService
from app.services.kandang_service import KandangService
from app.api.deps import get_current_user, get_iot_auth

router = APIRouter()


@router.post(
    "/iot",
    response_model=BaseResponse[SensorDataResponse],
    status_code=status.HTTP_201_CREATED,
    summary="IoT Device - Submit Sensor Data",
    description="Endpoint khusus perangkat IoT (ESP32). Auth via X-API-Key header, bukan JWT.",
)
async def create_sensor_data_iot(
    data: SensorDataIoTCreate,
    _: str = Depends(get_iot_auth),
    db: AsyncSession = Depends(get_db),
):
    """
    Terima data sensor dari perangkat IoT (ESP32).

    - Auth: header `X-API-Key` (bukan Bearer token)
    - `hari_ke` dihitung otomatis dari data pertama kandang
    - Otomatis trigger ML classification & forecasting
    - Otomatis broadcast ke WebSocket dashboard
    """
    from app.ml.model_loader import predict_classification, predict_forecasting
    from app.services.notification_service import NotificationService
    from app.services.prediction_service import PredictionService
    import logging

    logger = logging.getLogger(__name__)

    kandang_service = KandangService(db)
    kandang = await kandang_service.get_by_id(data.kandang_id)
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan",
        )

    sensor_service = SensorDataService(db)
    hari_ke = await sensor_service.get_hari_ke(data.kandang_id)

    sensor_create = SensorDataCreate(
        kandang_id=data.kandang_id,
        timestamp=datetime.now(),
        hari_ke=hari_ke,
        suhu=data.temperature,
        kelembaban=data.humidity,
        amoniak=data.ammonia,
    )
    sensor_data = await sensor_service.create(sensor_create)

    # Auto ML predictions
    auto_prediction_results = {}
    try:
        features = {
            'Hari Ke-': hari_ke,
            'Suhu': data.temperature,
            'Kelembaban': data.humidity,
            'Amoniak': data.ammonia,
            'Pakan': 0,
            'Minum': 0,
            'Bobot': 0,
            'Populasi': 0,
            'Luas Kandang': kandang.kapasitas / 10 if kandang.kapasitas else 120,
            'Hour': sensor_data.timestamp.hour,
        }
        cls_result = predict_classification(features)
        auto_prediction_results["classification"] = {
            "prediction": cls_result['class'],
            "confidence": cls_result['confidence'],
            "is_abnormal": cls_result['class'] == 'Abnormal',
        }

        # Simpan hasil prediksi ke tabel predictions
        try:
            prediction_svc = PredictionService(db)
            await prediction_svc.save_classification(
                kandang_id=data.kandang_id,
                prediction=cls_result['class'],
                confidence=cls_result['confidence'],
                input_data={
                    "suhu": data.temperature,
                    "kelembaban": data.humidity,
                    "amoniak": data.ammonia,
                    "hari_ke": hari_ke,
                    "source": "iot_auto",
                },
                sensor_data_id=sensor_data.id,
            )
        except Exception as log_err:
            logger.warning(f"IoT classification save failed: {log_err}")

        if cls_result['class'] == 'Abnormal':
            try:
                notification_service = NotificationService(db)
                await notification_service.create_classification_alert(
                    user_id=kandang.pemilik_id,
                    kandang_id=data.kandang_id,
                    prediction=cls_result['class'],
                    confidence=cls_result['confidence'],
                    sensor_data=features,
                )
            except Exception as notif_err:
                logger.warning(f"IoT auto-classification notification failed: {notif_err}")
    except Exception as e:
        logger.warning(f"IoT auto-classification failed: {e}")
        auto_prediction_results["classification"] = {"error": str(e)}

    try:
        # Ambil 4 data dengan jarak ~30 menit (sesuai interval training ML)
        # Meski ESP32 kirim tiap 5 detik, forecasting tetap pakai data 30-menit
        recent_data = await sensor_service.get_latest_at_interval(
            kandang_id=data.kandang_id, n_points=4, interval_minutes=30
        )
        if len(recent_data) >= 4:
            sensor_history = [
                {'temp': sd.suhu, 'hum': sd.kelembaban, 'ammo': sd.amoniak, 'Death': sd.death or 0}
                for sd in recent_data[-4:]
            ]
            fc_result = predict_forecasting(sensor_history)
            auto_prediction_results["forecasting"] = {
                "predicted_death": fc_result['predicted_death'],
                "has_risk": fc_result['predicted_death'] > 0,
            }

            # Simpan hasil forecasting ke tabel predictions
            try:
                prediction_svc = PredictionService(db)
                await prediction_svc.save_forecasting(
                    kandang_id=data.kandang_id,
                    predicted_death=fc_result['predicted_death'],
                    raw_prediction=fc_result['raw_prediction'],
                    input_data={"sensor_history": sensor_history, "source": "iot_auto"},
                    sensor_data_id=sensor_data.id,
                )
            except Exception as fc_save_err:
                logger.warning(f"IoT forecasting save failed: {fc_save_err}")

            if fc_result['predicted_death'] > 1:
                try:
                    notification_service = NotificationService(db)
                    await notification_service.create_death_forecast_alert(
                        user_id=kandang.pemilik_id,
                        kandang_id=data.kandang_id,
                        predicted_death=fc_result['predicted_death'],
                        raw_prediction=fc_result['raw_prediction'],
                    )
                except Exception as notif_err:
                    logger.warning(f"IoT auto-forecast notification failed: {notif_err}")
        else:
            auto_prediction_results["forecasting"] = {
                "skipped": True,
                "reason": f"Need 4 data points x 30 menit, baru ada {len(recent_data)}",
            }
    except Exception as e:
        logger.warning(f"IoT auto-forecasting failed: {e}")
        auto_prediction_results["forecasting"] = {"error": str(e)}

    # WebSocket broadcast
    try:
        from app.services.notification_service import manager
        await manager.broadcast_all({
            "type": "sensor_data",
            "data": {
                "id": str(sensor_data.id),
                "kandang_id": str(sensor_data.kandang_id),
                "timestamp": sensor_data.timestamp.isoformat(),
                "hari_ke": sensor_data.hari_ke,
                "suhu": sensor_data.suhu,
                "kelembaban": sensor_data.kelembaban,
                "amoniak": sensor_data.amoniak,
                "auto_prediction": auto_prediction_results,
            },
        })
    except Exception as ws_err:
        logger.warning(f"IoT WebSocket broadcast failed: {ws_err}")

    return success_response(
        data=SensorDataResponse.model_validate(sensor_data),
        message="Data IoT berhasil disimpan",
    )


@router.post(
    "",
    response_model=BaseResponse[SensorDataResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create Sensor Data",
    description="Record new sensor data reading (from IoT device or manual input). Auto-triggers ML predictions.",
)
async def create_sensor_data(
    data: SensorDataCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new sensor data record.
    
    Can be called by IoT devices or users to record sensor readings.
    Automatically triggers ML classification and forecasting predictions.
    """
    from app.ml.model_loader import predict_classification, predict_forecasting
    from app.services.notification_service import NotificationService
    from app.services.activity_log_service import ActivityLogService
    from app.api.deps import get_request_info
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Verify kandang exists and user has access
    kandang_service = KandangService(db)
    kandang = await kandang_service.get_by_id(data.kandang_id)
    
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan"
        )
    
    # Check access (admin can access all, others only their kandang)
    if current_user.role != "admin" and kandang.pemilik_id != current_user.id:
        # Check if user is a peternak assigned to this kandang's owner
        if current_user.pemilik_id != kandang.pemilik_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tidak memiliki akses ke kandang ini"
            )
    
    sensor_service = SensorDataService(db)
    sensor_data = await sensor_service.create(data, recorded_by=current_user.id)
    
    # ============================================
    # AUTO-PREDICTION: Classification + Forecasting
    # ============================================
    auto_prediction_results = {}
    
    try:
        # --- Auto Classification ---
        features = {
            'Hari Ke-': data.hari_ke,
            'Suhu': data.suhu,
            'Kelembaban': data.kelembaban,
            'Amoniak': data.amoniak,
            'Pakan': data.pakan or 0,
            'Minum': data.minum or 0,
            'Bobot': data.bobot or 0,
            'Populasi': data.populasi or 0,
            'Luas Kandang': kandang.kapasitas / 10 if kandang.kapasitas else 120,
            'Hour': sensor_data.timestamp.hour if sensor_data.timestamp else 12,
        }
        
        cls_result = predict_classification(features)
        auto_prediction_results["classification"] = {
            "prediction": cls_result['class'],
            "confidence": cls_result['confidence'],
            "is_abnormal": cls_result['class'] == 'Abnormal',
        }
        
        # Notify if abnormal
        if cls_result['class'] == 'Abnormal':
            try:
                notification_service = NotificationService(db)
                await notification_service.create_classification_alert(
                    user_id=current_user.id,
                    kandang_id=data.kandang_id,
                    prediction=cls_result['class'],
                    confidence=cls_result['confidence'],
                    sensor_data=features,
                )
            except Exception as notif_err:
                logger.warning(f"Auto-classification notification failed: {notif_err}")
        
        logger.info(f"Auto-classification: {cls_result['class']} ({cls_result['confidence']:.1%})")
        
    except Exception as e:
        logger.warning(f"Auto-classification failed: {e}")
        auto_prediction_results["classification"] = {"error": str(e)}
    
    try:
        # --- Auto Forecasting (need 4+ data points) ---
        recent_data = await sensor_service.get_latest(
            kandang_id=data.kandang_id, limit=4
        )
        
        if len(recent_data) >= 4:
            sensor_history = [
                {
                    'temp': sd.suhu,
                    'hum': sd.kelembaban,
                    'ammo': sd.amoniak,
                    'Death': sd.death or 0,
                }
                for sd in recent_data[-4:]  # oldest first
            ]
            
            fc_result = predict_forecasting(sensor_history)
            auto_prediction_results["forecasting"] = {
                "predicted_death": fc_result['predicted_death'],
                "has_risk": fc_result['predicted_death'] > 0,
            }
            
            # Notify if significant death predicted (>1, since 1 may be model noise)
            if fc_result['predicted_death'] > 1:
                try:
                    notification_service = NotificationService(db)
                    await notification_service.create_death_forecast_alert(
                        user_id=current_user.id,
                        kandang_id=data.kandang_id,
                        predicted_death=fc_result['predicted_death'],
                        raw_prediction=fc_result['raw_prediction'],
                    )
                except Exception as notif_err:
                    logger.warning(f"Auto-forecast notification failed: {notif_err}")
            
            logger.info(f"Auto-forecast: {fc_result['predicted_death']} deaths predicted")
        else:
            auto_prediction_results["forecasting"] = {
                "skipped": True,
                "reason": f"Need 4+ data points, have {len(recent_data)}",
            }
            
    except Exception as e:
        logger.warning(f"Auto-forecasting failed: {e}")
        auto_prediction_results["forecasting"] = {"error": str(e)}
    
    # ============================================
    # BROADCAST: Push new sensor data to all WebSocket clients
    # ============================================
    try:
        from app.services.notification_service import manager
        await manager.broadcast_all({
            "type": "sensor_data",
            "data": {
                "id": str(sensor_data.id),
                "kandang_id": str(sensor_data.kandang_id),
                "timestamp": sensor_data.timestamp.isoformat() if sensor_data.timestamp else None,
                "hari_ke": sensor_data.hari_ke,
                "suhu": sensor_data.suhu,
                "kelembaban": sensor_data.kelembaban,
                "amoniak": sensor_data.amoniak,
                "pakan": sensor_data.pakan,
                "minum": sensor_data.minum,
                "populasi": sensor_data.populasi,
                "bobot": sensor_data.bobot,
                "death": sensor_data.death,
                "auto_prediction": auto_prediction_results,
            },
        })
    except Exception as ws_err:
        logger.warning(f"WebSocket broadcast failed: {ws_err}")
    
    return success_response(
        data=SensorDataResponse.model_validate(sensor_data),
        message="Data sensor berhasil disimpan (auto-prediction triggered)",
    )


@router.put(
    "/{sensor_data_id}/manual",
    response_model=BaseResponse[SensorDataResponse],
    summary="Update Manual Fields",
    description="Update manually-input fields (pakan, minum, bobot, populasi, death)",
)
async def update_manual_fields(
    sensor_data_id: uuid.UUID,
    data: SensorDataManualUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update manual input fields for an existing sensor data record.
    
    Used by peternak/pemilik to add manually-collected data.
    """
    sensor_service = SensorDataService(db)
    sensor_data = await sensor_service.get_by_id(sensor_data_id)
    
    if not sensor_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data sensor tidak ditemukan"
        )
    
    # Verify access via kandang
    kandang_service = KandangService(db)
    kandang = await kandang_service.get_by_id(sensor_data.kandang_id)
    
    if current_user.role != "admin" and kandang.pemilik_id != current_user.id:
        if current_user.pemilik_id != kandang.pemilik_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tidak memiliki akses untuk update data ini"
            )
    
    updated = await sensor_service.update_manual_fields(
        sensor_data_id, data, current_user.id
    )
    
    return success_response(
        data=SensorDataResponse.model_validate(updated),
        message="Data manual berhasil diupdate",
    )


@router.get(
    "/kandang/{kandang_id}",
    response_model=BaseResponse[SensorDataListResponse],
    summary="Get Sensor Data by Kandang",
    description="Get paginated sensor data for a specific kandang",
)
async def get_sensor_data_by_kandang(
    kandang_id: uuid.UUID,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=100),
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get sensor data for a kandang with pagination and optional time filtering.
    """
    # Verify kandang access
    kandang_service = KandangService(db)
    kandang = await kandang_service.get_by_id(kandang_id)
    
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan"
        )
    
    if current_user.role != "admin" and kandang.pemilik_id != current_user.id:
        if current_user.pemilik_id != kandang.pemilik_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tidak memiliki akses ke kandang ini"
            )
    
    sensor_service = SensorDataService(db)
    items, total = await sensor_service.get_by_kandang(
        kandang_id, page, page_size, start_time, end_time
    )
    
    return success_response(
        data=SensorDataListResponse(
            items=[SensorDataResponse.model_validate(item) for item in items],
            total=total,
            page=page,
            page_size=page_size,
        ),
        message="Data sensor berhasil diambil",
    )


@router.get(
    "/kandang/{kandang_id}/latest",
    response_model=BaseResponse[list[SensorDataResponse]],
    summary="Get Latest Sensor Data",
    description="Get latest N sensor data readings for a kandang (default 4)",
)
async def get_latest_sensor_data(
    kandang_id: uuid.UUID,
    limit: int = Query(default=4, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get latest sensor data for forecasting or display.
    Returns data in chronological order (oldest first).
    """
    # Verify kandang access
    kandang_service = KandangService(db)
    kandang = await kandang_service.get_by_id(kandang_id)
    
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan"
        )
    
    if current_user.role != "admin" and kandang.pemilik_id != current_user.id:
        if current_user.pemilik_id != kandang.pemilik_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tidak memiliki akses ke kandang ini"
            )
    
    sensor_service = SensorDataService(db)
    items = await sensor_service.get_latest(kandang_id, limit)
    
    return success_response(
        data=[SensorDataResponse.model_validate(item) for item in items],
        message=f"{len(items)} data sensor terbaru",
    )


@router.get(
    "/kandang/{kandang_id}/stats",
    response_model=BaseResponse[SensorDataStats],
    summary="Get Sensor Statistics",
    description="Get statistical summary of sensor data for a kandang",
)
async def get_sensor_stats(
    kandang_id: uuid.UUID,
    hours: int = Query(default=24, ge=1, le=168),  # Max 1 week
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get statistical summary (averages, totals) for the specified time period.
    """
    # Verify kandang access
    kandang_service = KandangService(db)
    kandang = await kandang_service.get_by_id(kandang_id)
    
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan"
        )
    
    if current_user.role != "admin" and kandang.pemilik_id != current_user.id:
        if current_user.pemilik_id != kandang.pemilik_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Tidak memiliki akses ke kandang ini"
            )
    
    sensor_service = SensorDataService(db)
    stats = await sensor_service.get_stats(kandang_id, hours)
    
    if not stats:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tidak ada data sensor dalam periode yang diminta"
        )
    
    return success_response(
        data=stats,
        message=f"Statistik {hours} jam terakhir",
    )


@router.delete(
    "/{sensor_data_id}",
    response_model=BaseResponse[dict],
    summary="Delete Sensor Data",
    description="Delete a sensor data record (Admin only)",
)
async def delete_sensor_data(
    sensor_data_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a sensor data record. Admin only.
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Hanya admin yang dapat menghapus data sensor"
        )
    
    sensor_service = SensorDataService(db)
    deleted = await sensor_service.delete(sensor_data_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Data sensor tidak ditemukan"
        )
    
    return success_response(
        data={"deleted_id": str(sensor_data_id)},
        message="Data sensor berhasil dihapus",
    )
