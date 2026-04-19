import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User, UserRole
from app.models.kandang import Kandang
from app.schemas.kandang import KandangCreate, KandangUpdate, KandangResponse
from app.schemas.base import (
    BaseResponse,
    PaginatedResponse,
    success_response,
    paginated_response,
)
from app.services.kandang_service import KandangService
from app.services.user_service import UserService
from app.services.activity_log_service import ActivityLogService
from app.api.deps import get_current_user, require_admin, require_pemilik_or_admin, get_request_info

router = APIRouter()


@router.get(
    "/me",
    response_model=BaseResponse[KandangResponse],
    summary="Get Kandang",
    description="Get detail kandang aktif",
)
async def get_my_kandang(
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    kandang_service = KandangService(db)
    result = await db.execute(
        select(Kandang)
        .options(selectinload(Kandang.pemilik))
        .where(Kandang.is_active == True)
        .limit(1)
    )
    kandang = result.scalar_one_or_none()
    if not kandang:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kandang tidak ditemukan")

    response = KandangResponse.model_validate(kandang)
    if kandang.pemilik:
        response.pemilik_name = kandang.pemilik.full_name
    response.latest_sensor = await kandang_service.get_latest_sensor(kandang.id)

    return success_response(data=response, message="Data kandang berhasil diambil")


@router.get(
    "",
    include_in_schema=False,
    response_model=PaginatedResponse[KandangResponse],
    summary="List Kandangs",
    description="Get paginated list of kandangs",
)
async def list_kandangs(
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(10, ge=1, le=100, description="Items per page"),
    pemilik_id: Optional[uuid.UUID] = Query(None, description="Filter by pemilik"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    search: Optional[str] = Query(None, description="Search by name or code"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get list of kandangs with pagination and filters."""
    kandang_service = KandangService(db)
    
    # Non-admin users can only see their own kandangs or kandangs of their pemilik
    filter_pemilik_id = pemilik_id
    
    if current_user.role == UserRole.PEMILIK:
        # Pemilik can only see their own kandangs
        filter_pemilik_id = current_user.id
    elif current_user.role == UserRole.PETERNAK:
        # Peternak can only see kandangs of their pemilik
        if current_user.pemilik_id:
            filter_pemilik_id = current_user.pemilik_id
        else:
            # Peternak without pemilik can't see any kandangs
            return paginated_response(
                data=[],
                total=0,
                page=page,
                per_page=per_page,
                message="Daftar kandang berhasil diambil",
            )
    
    kandangs, total = await kandang_service.get_list(
        page=page,
        per_page=per_page,
        pemilik_id=filter_pemilik_id,
        is_active=is_active,
        search=search,
    )
    
    kandang_responses = []
    for k in kandangs:
        response = KandangResponse.model_validate(k)
        if k.pemilik:
            response.pemilik_name = k.pemilik.full_name
        response.latest_sensor = await kandang_service.get_latest_sensor(k.id)
        kandang_responses.append(response)

    return paginated_response(
        data=kandang_responses,
        total=total,
        page=page,
        per_page=per_page,
        message="Daftar kandang berhasil diambil",
    )


@router.get(
    "/{kandang_id}",
    include_in_schema=False,
    response_model=BaseResponse[KandangResponse],
    summary="Get Kandang",
    description="Get kandang by ID",
)
async def get_kandang(
    kandang_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get kandang by ID."""
    kandang_service = KandangService(db)
    
    kandang = await kandang_service.get_by_id(kandang_id)
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan",
        )
    
    # Check access
    if current_user.role == UserRole.PEMILIK:
        if kandang.pemilik_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Akses ditolak",
            )
    elif current_user.role == UserRole.PETERNAK:
        if kandang.pemilik_id != current_user.pemilik_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Akses ditolak",
            )
    
    response = KandangResponse.model_validate(kandang)
    if kandang.pemilik:
        response.pemilik_name = kandang.pemilik.full_name
    response.latest_sensor = await kandang_service.get_latest_sensor(kandang.id)

    return success_response(
        data=response,
        message="Data kandang berhasil diambil",
    )


@router.post(
    "",
    include_in_schema=False,
    response_model=BaseResponse[KandangResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create Kandang",
    description="Create new kandang (Admin or Pemilik)",
)
async def create_kandang(
    request: Request,
    kandang_data: KandangCreate,
    current_user: User = Depends(require_pemilik_or_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Create a new kandang."""
    kandang_service = KandangService(db)
    user_service = UserService(db)
    
    # Validate pemilik_id
    pemilik = await user_service.get_by_id(kandang_data.pemilik_id)
    if not pemilik or pemilik.role != UserRole.PEMILIK:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pemilik tidak valid",
        )
    
    # Pemilik can only create kandang for themselves
    if current_user.role == UserRole.PEMILIK:
        if kandang_data.pemilik_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Pemilik hanya dapat membuat kandang untuk diri sendiri",
            )
    
    # Check if kode already exists
    existing = await kandang_service.get_by_kode(kandang_data.kode)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kode kandang sudah terdaftar",
        )
    
    kandang = await kandang_service.create(kandang_data)
    
    # Log activity
    request_info = await get_request_info(request)
    activity_service = ActivityLogService(db)
    await activity_service.log_action(
        user_id=current_user.id,
        action="create_kandang",
        resource="kandang",
        resource_id=kandang.id,
        request_info=request_info,
        details={"kode": kandang.kode, "nama": kandang.nama},
    )
    
    response = KandangResponse.model_validate(kandang)
    
    return success_response(
        data=response,
        message="Kandang berhasil dibuat",
    )


@router.put(
    "/{kandang_id}",
    include_in_schema=False,
    response_model=BaseResponse[KandangResponse],
    summary="Update Kandang",
    description="Update kandang data",
)
async def update_kandang(
    kandang_id: uuid.UUID,
    request: Request,
    kandang_data: KandangUpdate,
    current_user: User = Depends(require_pemilik_or_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Update kandang data."""
    kandang_service = KandangService(db)
    
    kandang = await kandang_service.get_by_id(kandang_id)
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan",
        )
    
    # Pemilik can only update their own kandangs
    if current_user.role == UserRole.PEMILIK:
        if kandang.pemilik_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Akses ditolak",
            )
    
    updated = await kandang_service.update(kandang, kandang_data)
    
    # Log activity
    request_info = await get_request_info(request)
    activity_service = ActivityLogService(db)
    await activity_service.log_action(
        user_id=current_user.id,
        action="update_kandang",
        resource="kandang",
        resource_id=kandang_id,
        request_info=request_info,
        details={"updated_fields": list(kandang_data.model_dump(exclude_unset=True).keys())},
    )
    
    response = KandangResponse.model_validate(updated)
    
    return success_response(
        data=response,
        message="Kandang berhasil diupdate",
    )


@router.delete(
    "/{kandang_id}",
    include_in_schema=False,
    response_model=BaseResponse,
    summary="Delete Kandang",
    description="Soft delete (deactivate) a kandang",
)
async def delete_kandang(
    kandang_id: uuid.UUID,
    request: Request,
    current_user: User = Depends(require_pemilik_or_admin()),
    db: AsyncSession = Depends(get_db),
):
    """Soft delete (deactivate) a kandang."""
    kandang_service = KandangService(db)
    
    kandang = await kandang_service.get_by_id(kandang_id)
    if not kandang:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Kandang tidak ditemukan",
        )
    
    # Pemilik can only delete their own kandangs
    if current_user.role == UserRole.PEMILIK:
        if kandang.pemilik_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Akses ditolak",
            )
    
    await kandang_service.deactivate(kandang)
    
    # Log activity
    request_info = await get_request_info(request)
    activity_service = ActivityLogService(db)
    await activity_service.log_action(
        user_id=current_user.id,
        action="delete_kandang",
        resource="kandang",
        resource_id=kandang_id,
        request_info=request_info,
        details={"kode": kandang.kode},
    )
    
    return success_response(message="Kandang berhasil dihapus")
