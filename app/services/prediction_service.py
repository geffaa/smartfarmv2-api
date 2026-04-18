"""
Prediction Service - Simpan dan ambil hasil prediksi ML
"""
import uuid
import json
from typing import Optional, List

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.prediction import Prediction


class PredictionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def save_classification(
        self,
        kandang_id: uuid.UUID,
        prediction: str,
        confidence: float,
        input_data: dict,
        sensor_data_id: Optional[uuid.UUID] = None,
    ) -> Prediction:
        record = Prediction(
            kandang_id=kandang_id,
            sensor_data_id=sensor_data_id,
            type="classification",
            prediction=prediction,
            confidence=confidence,
            input_data=json.dumps(input_data),
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def save_forecasting(
        self,
        kandang_id: uuid.UUID,
        predicted_death: int,
        raw_prediction: float,
        input_data: dict,
        sensor_data_id: Optional[uuid.UUID] = None,
    ) -> Prediction:
        record = Prediction(
            kandang_id=kandang_id,
            sensor_data_id=sensor_data_id,
            type="forecasting",
            predicted_death=predicted_death,
            raw_prediction=raw_prediction,
            input_data=json.dumps(input_data),
        )
        self.db.add(record)
        await self.db.commit()
        await self.db.refresh(record)
        return record

    async def get_history(
        self,
        kandang_id: uuid.UUID,
        limit: int = 50,
        prediction_type: Optional[str] = None,
    ) -> List[Prediction]:
        query = select(Prediction).where(Prediction.kandang_id == kandang_id)
        if prediction_type:
            query = query.where(Prediction.type == prediction_type)
        query = query.order_by(desc(Prediction.created_at)).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())
