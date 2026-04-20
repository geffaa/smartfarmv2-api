"""
Notification Service - Business logic for notifications with WebSocket broadcast
and optional WhatsApp via Fonnte (fonnte.com).
"""
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict

from sqlalchemy import select, func, and_, desc, update
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import WebSocket

from app.models.notification import Notification, NotificationType
from app.models.user import User
from app.schemas.notification import NotificationCreate
from app.services.fonnte_service import (
    send_whatsapp,
    build_abnormal_message,
    build_death_forecast_message,
)


class ConnectionManager:
    """WebSocket connection manager for real-time notifications."""
    
    def __init__(self):
        # Map of user_id -> list of active websocket connections
        self.active_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept a new WebSocket connection for a user."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        print(f"🔗 WebSocket connected for user {user_id}")
    
    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        print(f"🔌 WebSocket disconnected for user {user_id}")
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send a message to all connections of a specific user."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    print(f"Error sending to user {user_id}: {e}")
    
    async def broadcast_to_users(self, user_ids: List[str], message: dict):
        """Broadcast a message to multiple users."""
        for user_id in user_ids:
            await self.send_to_user(user_id, message)
    
    async def broadcast_all(self, message: dict):
        """Broadcast a message to ALL connected users."""
        for user_id in list(self.active_connections.keys()):
            await self.send_to_user(user_id, message)


# Global connection manager instance
manager = ConnectionManager()


class NotificationService:
    """Service class for notification operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create(
        self,
        data: NotificationCreate,
        broadcast: bool = True
    ) -> Notification:
        """Create a new notification and optionally broadcast via WebSocket."""
        notification = Notification(
            user_id=data.user_id,
            kandang_id=data.kandang_id,
            type=data.type,
            title=data.title,
            message=data.message,
            data=data.data,
        )
        self.db.add(notification)
        await self.db.commit()
        await self.db.refresh(notification)
        
        # Broadcast via WebSocket if enabled
        if broadcast:
            await self._broadcast_notification(notification)
        
        return notification
    
    async def _broadcast_notification(self, notification: Notification):
        """Broadcast notification to user via WebSocket."""
        message = {
            "type": "notification",
            "data": {
                "id": str(notification.id),
                "notification_type": notification.type,
                "title": notification.title,
                "message": notification.message,
                "kandang_id": str(notification.kandang_id) if notification.kandang_id else None,
                "created_at": notification.created_at.isoformat(),
            }
        }
        await manager.send_to_user(str(notification.user_id), message)
    
    async def get_by_user(
        self,
        user_id: uuid.UUID,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Notification], int, int]:
        """Get notifications for a user.
        
        Returns: (notifications, total_count, unread_count)
        """
        query = select(Notification).where(Notification.user_id == user_id)
        
        if unread_only:
            query = query.where(Notification.is_read == False)
        
        # Count total
        count_query = select(func.count()).select_from(
            select(Notification).where(Notification.user_id == user_id).subquery()
        )
        total_result = await self.db.execute(count_query)
        total = total_result.scalar()
        
        # Count unread
        unread_query = select(func.count()).where(
            and_(
                Notification.user_id == user_id,
                Notification.is_read == False
            )
        )
        unread_result = await self.db.execute(unread_query)
        unread_count = unread_result.scalar()
        
        # Get notifications
        query = query.order_by(desc(Notification.created_at)).offset(offset).limit(limit)
        result = await self.db.execute(query)
        items = list(result.scalars().all())

        return items, total, unread_count
    
    async def get_unread_count(self, user_id: uuid.UUID) -> int:
        """Get count of unread notifications for a user."""
        result = await self.db.execute(
            select(func.count()).where(
                and_(
                    Notification.user_id == user_id,
                    Notification.is_read == False
                )
            )
        )
        return result.scalar()
    
    async def mark_as_read(self, notification_id: uuid.UUID, user_id: uuid.UUID) -> Optional[Notification]:
        """Mark a notification as read."""
        result = await self.db.execute(
            select(Notification).where(
                and_(
                    Notification.id == notification_id,
                    Notification.user_id == user_id
                )
            )
        )
        notification = result.scalar_one_or_none()
        
        if not notification:
            return None
        
        notification.is_read = True
        notification.read_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(notification)
        return notification
    
    async def mark_all_as_read(self, user_id: uuid.UUID) -> int:
        """Mark all notifications as read for a user. Returns count of updated."""
        result = await self.db.execute(
            update(Notification)
            .where(
                and_(
                    Notification.user_id == user_id,
                    Notification.is_read == False
                )
            )
            .values(is_read=True, read_at=datetime.utcnow())
        )
        await self.db.commit()
        return result.rowcount
    
    async def _get_user_phone(self, user_id: uuid.UUID) -> Optional[str]:
        """Fetch user's phone number from DB."""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        return user.phone if user else None

    async def _get_kandang_name(self, kandang_id: uuid.UUID) -> str:
        """Fetch kandang name from DB."""
        from app.models.kandang import Kandang
        result = await self.db.execute(
            select(Kandang).where(Kandang.id == kandang_id)
        )
        kandang = result.scalar_one_or_none()
        return kandang.nama if kandang else "Kandang"

    async def create_classification_alert(
        self,
        user_id: uuid.UUID,
        kandang_id: uuid.UUID,
        prediction: str,
        sensor_data: dict,
    ):
        """Create in-app notification + kirim WhatsApp untuk kondisi Abnormal."""
        if prediction != "Abnormal":
            return None

        data = NotificationCreate(
            user_id=user_id,
            kandang_id=kandang_id,
            type=NotificationType.ABNORMAL_CLASSIFICATION.value,
            title="⚠️ Kondisi Abnormal Terdeteksi",
            message=(
                f"Sistem mendeteksi kondisi kandang tidak normal. "
                f"Suhu: {sensor_data.get('Suhu', 'N/A')}°C, "
                f"Kelembaban: {sensor_data.get('Kelembaban', 'N/A')}%, "
                f"Amoniak: {sensor_data.get('Amoniak', 'N/A')} ppm. "
                f"Segera periksa kondisi kandang!"
            ),
            data=json.dumps(sensor_data),
        )
        notification = await self.create(data)

        # WhatsApp via Fonnte (non-blocking — error tidak membatalkan flow)
        try:
            phone = await self._get_user_phone(user_id)
            if phone:
                kandang_name = await self._get_kandang_name(kandang_id)
                wa_message = build_abnormal_message(kandang_name, sensor_data)
                await send_whatsapp(phone, wa_message)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Fonnte classification alert failed: {e}")

        return notification

    async def create_death_forecast_alert(
        self,
        user_id: uuid.UUID,
        kandang_id: uuid.UUID,
        predicted_death: int,
        raw_prediction: float,
    ):
        """Create in-app notification + kirim WhatsApp untuk prediksi kematian > 1."""
        if predicted_death <= 1:
            return None

        data = NotificationCreate(
            user_id=user_id,
            kandang_id=kandang_id,
            type=NotificationType.DEATH_FORECAST.value,
            title=f"🚨 Prediksi {predicted_death} Kematian",
            message=(
                f"Sistem memprediksi {predicted_death} ekor ayam berisiko mati "
                f"dalam 30 menit ke depan. Segera periksa kondisi kandang!"
            ),
            data=json.dumps({
                "predicted_death": predicted_death,
                "raw_prediction": raw_prediction,
            }),
        )
        notification = await self.create(data)

        # WhatsApp via Fonnte
        try:
            phone = await self._get_user_phone(user_id)
            if phone:
                kandang_name = await self._get_kandang_name(kandang_id)
                wa_message = build_death_forecast_message(kandang_name, predicted_death)
                await send_whatsapp(phone, wa_message)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Fonnte forecast alert failed: {e}")

        return notification
