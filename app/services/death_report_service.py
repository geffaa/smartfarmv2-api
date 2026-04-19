import uuid
from datetime import datetime, timedelta, date
from typing import Optional, List

from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.death_report import DeathReport
from app.schemas.death_report import DeathReportCreate


class DeathReportService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create(
        self,
        data: DeathReportCreate,
        kandang_id: uuid.UUID,
        reported_by: Optional[uuid.UUID] = None,
    ) -> DeathReport:
        report = DeathReport(
            kandang_id=kandang_id,
            count=data.count,
            notes=data.notes,
            timestamp=data.timestamp or datetime.utcnow(),
            reported_by=reported_by,
        )
        self.db.add(report)
        await self.db.commit()
        await self.db.refresh(report)
        return report

    async def get_total_today(self, kandang_id: uuid.UUID) -> int:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        result = await self.db.execute(
            select(func.coalesce(func.sum(DeathReport.count), 0))
            .where(
                and_(
                    DeathReport.kandang_id == kandang_id,
                    DeathReport.timestamp >= today_start,
                )
            )
        )
        return int(result.scalar() or 0)

    async def get_list(
        self,
        kandang_id: uuid.UUID,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[List[DeathReport], int]:
        count_result = await self.db.execute(
            select(func.count(DeathReport.id)).where(DeathReport.kandang_id == kandang_id)
        )
        total = count_result.scalar() or 0

        result = await self.db.execute(
            select(DeathReport)
            .where(DeathReport.kandang_id == kandang_id)
            .order_by(desc(DeathReport.timestamp))
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
        return list(result.scalars().all()), total
