"""
Scheduler Service - Penjadwalan tugas otomatis (APScheduler)
"""
import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone="Asia/Jakarta")


async def send_daily_log_reminder():
    """
    Kirim notifikasi pengingat input log harian setiap pagi jam 07:00 WIB.
    Dikirim ke pemilik + semua peternak aktif di bawahnya, via in-app + WhatsApp.
    Deduplication: skip jika notifikasi hari ini sudah pernah dikirim ke user tsb.
    """
    from app.database import async_session_maker as AsyncSessionLocal
    from app.models.kandang import Kandang
    from app.models.user import User, UserRole
    from app.models.notification import Notification, NotificationType
    from app.services.notification_service import NotificationService
    from app.services.fonnte_service import send_whatsapp
    from app.schemas.notification import NotificationCreate
    from sqlalchemy import select, func

    now = datetime.now()
    today_str = now.strftime("%d %b %Y")
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(Kandang).where(Kandang.is_active == True)
            )
            kandangs = result.scalars().all()

            for kandang in kandangs:
                try:
                    svc = NotificationService(db)

                    title = "📋 Pengingat Log Harian"
                    message = (
                        f"Selamat pagi! Jangan lupa mengisi log harian kandang "
                        f"*{kandang.nama}* hari ini ({today_str}). "
                        f"Catat pakan, minum, bobot, dan populasi ayam."
                    )
                    wa_msg = (
                        f"📋 *Pengingat Log Harian - Broilabs*\n\n"
                        f"Selamat pagi! 🌅\n"
                        f"Jangan lupa mengisi log harian kandang *{kandang.nama}* hari ini.\n\n"
                        f"Catat:\n"
                        f"• Jumlah pakan (kg)\n"
                        f"• Jumlah minum (liter)\n"
                        f"• Bobot rata-rata (gram)\n"
                        f"• Populasi saat ini\n\n"
                        f"🔗 https://broilabs.ukirbin.com/daily-logs"
                    )

                    # Pemilik + semua peternak aktif di bawah pemilik kandang
                    users_result = await db.execute(
                        select(User).where(
                            (User.id == kandang.pemilik_id) |
                            (
                                (User.pemilik_id == kandang.pemilik_id) &
                                (User.role == UserRole.PETERNAK) &
                                (User.is_active == True)
                            )
                        )
                    )
                    target_users = users_result.scalars().all()

                    sent_count = 0
                    for user in target_users:
                        # Deduplication: skip jika sudah ada notif hari ini untuk user+kandang ini
                        already_sent = await db.execute(
                            select(func.count()).where(
                                (Notification.user_id == user.id) &
                                (Notification.kandang_id == kandang.id) &
                                (Notification.type == NotificationType.SYSTEM.value) &
                                (Notification.title == title) &
                                (Notification.created_at >= today_start)
                            )
                        )
                        if already_sent.scalar() > 0:
                            continue

                        notif_data = NotificationCreate(
                            user_id=user.id,
                            kandang_id=kandang.id,
                            type=NotificationType.SYSTEM.value,
                            title=title,
                            message=message,
                        )
                        await svc.create(notif_data, broadcast=True)

                        if user.phone:
                            await send_whatsapp(user.phone, wa_msg)

                        sent_count += 1

                    logger.info(
                        f"Daily log reminder: kandang '{kandang.nama}' → "
                        f"{sent_count} sent, {len(target_users) - sent_count} skipped (already sent today)"
                    )
                except Exception as e:
                    logger.warning(f"Failed to send reminder for kandang {kandang.id}: {e}")

        except Exception as e:
            logger.error(f"Daily log reminder job failed: {e}")


async def check_iot_offline():
    """
    Cek setiap 30 menit apakah ada kandang aktif yang tidak mengirim data IoT
    lebih dari 35 menit. Jika ya, kirim notifikasi ke semua admin.
    Deduplication: skip jika notif offline sudah dikirim dalam 1 jam terakhir.
    """
    from app.database import async_session_maker as AsyncSessionLocal
    from app.models.kandang import Kandang
    from app.models.user import User, UserRole
    from app.models.notification import Notification, NotificationType
    from app.models.sensor_data import SensorData
    from app.services.notification_service import NotificationService
    from app.schemas.notification import NotificationCreate
    from sqlalchemy import select, func
    from datetime import timedelta

    now = datetime.utcnow()
    offline_threshold = now - timedelta(minutes=35)
    cooldown_threshold = now - timedelta(hours=1)

    async with AsyncSessionLocal() as db:
        try:
            kandangs_result = await db.execute(
                select(Kandang).where(Kandang.is_active == True)
            )
            kandangs = kandangs_result.scalars().all()

            admins_result = await db.execute(
                select(User).where(
                    (User.role == UserRole.ADMIN) &
                    (User.is_active == True)
                )
            )
            admins = admins_result.scalars().all()
            if not admins:
                return

            for kandang in kandangs:
                try:
                    # Cek data sensor terakhir
                    last_data_result = await db.execute(
                        select(SensorData.timestamp)
                        .where(SensorData.kandang_id == kandang.id)
                        .order_by(SensorData.timestamp.desc())
                        .limit(1)
                    )
                    last_ts = last_data_result.scalar_one_or_none()

                    # Kandang baru tanpa data sama sekali, atau data terakhir > 35 menit lalu
                    is_offline = (last_ts is None) or (last_ts.replace(tzinfo=None) < offline_threshold)
                    if not is_offline:
                        continue

                    last_seen = "Belum pernah ada data" if last_ts is None else \
                        f"{int((now - last_ts.replace(tzinfo=None)).total_seconds() // 60)} menit lalu"

                    title = f"⚠️ IoT Offline: {kandang.nama}"
                    message = (
                        f"Kandang *{kandang.nama}* tidak mengirim data sensor. "
                        f"Data terakhir: {last_seen}. Cek koneksi perangkat IoT."
                    )

                    svc = NotificationService(db)
                    for admin in admins:
                        # Cooldown: skip jika sudah kirim notif offline untuk kandang ini dalam 1 jam
                        already_sent = await db.execute(
                            select(func.count()).where(
                                (Notification.user_id == admin.id) &
                                (Notification.kandang_id == kandang.id) &
                                (Notification.type == NotificationType.SYSTEM.value) &
                                (Notification.title == title) &
                                (Notification.created_at >= cooldown_threshold)
                            )
                        )
                        if already_sent.scalar() > 0:
                            continue

                        notif_data = NotificationCreate(
                            user_id=admin.id,
                            kandang_id=kandang.id,
                            type=NotificationType.SYSTEM.value,
                            title=title,
                            message=message,
                        )
                        await svc.create(notif_data, broadcast=True)

                    logger.info(f"IoT offline alert sent for kandang '{kandang.nama}' to {len(admins)} admin(s)")
                except Exception as e:
                    logger.warning(f"IoT offline check failed for kandang {kandang.id}: {e}")

        except Exception as e:
            logger.error(f"IoT offline check job failed: {e}")


def start_scheduler():
    """Daftarkan jobs dan jalankan scheduler."""
    scheduler.add_job(
        send_daily_log_reminder,
        trigger=CronTrigger(hour=7, minute=0, timezone="Asia/Jakarta"),
        id="daily_log_reminder",
        name="Daily Log Reminder 07:00 WIB",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.add_job(
        check_iot_offline,
        trigger=CronTrigger(minute="*/30", timezone="Asia/Jakarta"),
        id="iot_offline_check",
        name="IoT Offline Check (every 30 min)",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=120,
    )
    scheduler.start()
    logger.info("Scheduler started — daily log reminder 07:00 WIB, IoT offline check every 30 min")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
