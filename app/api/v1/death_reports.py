from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.models.kandang import Kandang
from app.schemas.base import BaseResponse, success_response
from app.schemas.death_report import DeathReportCreate, DeathReportResponse, DeathReportListResponse
from app.services.death_report_service import DeathReportService
from app.api.deps import get_current_user, get_single_kandang

router = APIRouter()


@router.post(
    "",
    response_model=BaseResponse[DeathReportResponse],
    summary="Laporkan Kematian",
    description="Catat kejadian kematian ayam",
)
async def report_death(
    data: DeathReportCreate,
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    service = DeathReportService(db)
    report = await service.create(data, kandang_id=kandang.id, reported_by=current_user.id)
    return success_response(
        data=DeathReportResponse.model_validate(report),
        message=f"Kematian {report.count} ekor berhasil dicatat",
    )


@router.get(
    "",
    response_model=BaseResponse[DeathReportListResponse],
    summary="Riwayat Laporan Kematian",
)
async def list_death_reports(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    service = DeathReportService(db)
    items, total = await service.get_list(kandang.id, page, per_page)
    return success_response(
        data=DeathReportListResponse(
            items=[DeathReportResponse.model_validate(r) for r in items],
            total=total,
            page=page,
            per_page=per_page,
        ),
        message="Riwayat laporan kematian berhasil diambil",
    )


@router.get(
    "/today-total",
    response_model=BaseResponse[dict],
    summary="Total Kematian Hari Ini",
)
async def get_today_total(
    current_user: User = Depends(get_current_user),
    kandang: Kandang = Depends(get_single_kandang),
    db: AsyncSession = Depends(get_db),
):
    service = DeathReportService(db)
    total = await service.get_total_today(kandang.id)
    return success_response(data={"total": total}, message="Total kematian hari ini")
