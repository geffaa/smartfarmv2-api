from fastapi import APIRouter

from app.api.v1 import auth, users, activity_logs, kandangs, predictions, sensor_data, notifications

api_router = APIRouter()

# Include all routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
    include_in_schema=False,
)

api_router.include_router(
    activity_logs.router,
    prefix="/activity-logs",
    tags=["Activity Logs"],
)

api_router.include_router(
    kandangs.router,
    prefix="/kandangs",
    tags=["Kandangs"],
    include_in_schema=False,
)

api_router.include_router(
    predictions.router,
    prefix="/predictions",
    tags=["Predictions"],
)

api_router.include_router(
    sensor_data.router,
    prefix="/sensor-data",
    tags=["Sensor Data"],
)

api_router.include_router(
    notifications.router,
    prefix="/notifications",
    tags=["Notifications"],
)
