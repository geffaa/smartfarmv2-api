from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.kandang import Kandang
from app.schemas.base import BaseResponse, success_response
from app.schemas.daily_log import DailyLogCreate, DailyLogResponse, DailyLogListResponse
from app.services.daily_log_service import DailyLogService
from app.api.deps import get_current_user, get_single_kandang

router = APIRouter()


@router.post(
    "",
    response_model=BaseResponse[DailyLogResponse],
    summary="Input Log Harian",
    description="Catat pakan, minum, populasi, bobot harian. Upsert berdasarkan tanggal.",
)
async def create_daily_log(
    data: DailyLogCreate,
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    service = DailyLogService(db)
    log = await service.create_or_update(data, kandang_id=kandang.id, recorded_by=current_user.id)
    return success_response(
        data=DailyLogResponse.model_validate(log),
        message="Log harian berhasil disimpan",
    )


@router.get(
    "/today",
    response_model=BaseResponse[DailyLogResponse],
    summary="Log Harian Hari Ini",
)
async def get_today_log(
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    service = DailyLogService(db)
    log = await service.get_today(kandang.id)
    if not log:
        return success_response(data=None, message="Belum ada log hari ini")
    return success_response(
        data=DailyLogResponse.model_validate(log),
        message="Log harian hari ini berhasil diambil",
    )


@router.get(
    "",
    response_model=BaseResponse[DailyLogListResponse],
    summary="Riwayat Log Harian",
)
async def list_daily_logs(
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    service = DailyLogService(db)
    items, total = await service.get_list(kandang.id, page, per_page)
    return success_response(
        data=DailyLogListResponse(
            items=[DailyLogResponse.model_validate(item) for item in items],
            total=total,
            page=page,
            per_page=per_page,
        ),
        message="Riwayat log harian berhasil diambil",
    )
