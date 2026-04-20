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
    Dikirim ke semua pemilik kandang yang aktif via in-app + WhatsApp.
    """
    from app.database import async_session_maker as AsyncSessionLocal
    from app.models.kandang import Kandang
    from app.models.user import User
    from app.services.notification_service import NotificationService
    from app.services.fonnte_service import send_whatsapp
    from app.schemas.notification import NotificationCreate
    from app.models.notification import NotificationType
    from sqlalchemy import select

    now = datetime.now().strftime("%d %b %Y")

    async with AsyncSessionLocal() as db:
        try:
            result = await db.execute(
                select(Kandang).where(Kandang.is_active == True)
            )
            kandangs = result.scalars().all()

            for kandang in kandangs:
                try:
                    svc = NotificationService(db)

                    notif_data = NotificationCreate(
                        user_id=kandang.pemilik_id,
                        kandang_id=kandang.id,
                        type=NotificationType.SYSTEM.value,
                        title="📋 Pengingat Log Harian",
                        message=(
                            f"Selamat pagi! Jangan lupa mengisi log harian kandang "
                            f"*{kandang.nama}* hari ini ({now}). "
                            f"Catat pakan, minum, bobot, dan populasi ayam."
                        ),
                    )
                    await svc.create(notif_data, broadcast=True)

                    # WhatsApp reminder
                    user_result = await db.execute(
                        select(User).where(User.id == kandang.pemilik_id)
                    )
                    user = user_result.scalar_one_or_none()
                    if user and user.phone:
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
                        await send_whatsapp(user.phone, wa_msg)

                    logger.info(f"Daily log reminder sent for kandang {kandang.nama}")
                except Exception as e:
                    logger.warning(f"Failed to send reminder for kandang {kandang.id}: {e}")

        except Exception as e:
            logger.error(f"Daily log reminder job failed: {e}")


def start_scheduler():
    """Daftarkan jobs dan jalankan scheduler."""
    scheduler.add_job(
        send_daily_log_reminder,
        trigger=CronTrigger(hour=7, minute=0, timezone="Asia/Jakarta"),
        id="daily_log_reminder",
        name="Daily Log Reminder 07:00 WIB",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("Scheduler started — daily log reminder at 07:00 WIB")


def stop_scheduler():
    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")
