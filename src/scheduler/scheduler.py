"""
APScheduler configuration for periodic jobs.
"""

from typing import Optional
from ..utils.logger import get_logger

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
except ImportError:
    # Fallback if apscheduler not installed
    AsyncIOScheduler = None
    IntervalTrigger = None
    CronTrigger = None

logger = get_logger("scheduler")


class AirGuardScheduler:
    """Scheduler for periodic AirGuard tasks."""
    
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
    
    def start(self):
        """Start the scheduler."""
        if self.scheduler is None:
            self.scheduler = AsyncIOScheduler()
            self.scheduler.start()
            logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if self.scheduler:
            self.scheduler.shutdown()
            self.scheduler = None
            logger.info("Scheduler stopped")
    
    def add_interval_job(self, func, seconds: int, id: str = None, **kwargs):
        """Add a job that runs at fixed intervals."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not started. Call start() first.")
        if IntervalTrigger is None:
            raise RuntimeError("APScheduler not installed. Install with: pip install apscheduler")
        
        trigger = IntervalTrigger(seconds=seconds)
        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=id or f"{func.__name__}_interval",
            **kwargs
        )
        logger.info(f"Added interval job: {id or func.__name__} (every {seconds}s)")
    
    def add_cron_job(self, func, hour: int = None, minute: int = None, id: str = None, **kwargs):
        """Add a job that runs on a cron schedule."""
        if not self.scheduler:
            raise RuntimeError("Scheduler not started. Call start() first.")
        if CronTrigger is None:
            raise RuntimeError("APScheduler not installed. Install with: pip install apscheduler")
        
        trigger = CronTrigger(hour=hour, minute=minute)
        self.scheduler.add_job(
            func,
            trigger=trigger,
            id=id or f"{func.__name__}_cron",
            **kwargs
        )
        logger.info(f"Added cron job: {id or func.__name__} (hour={hour}, minute={minute})")


# Global scheduler instance
scheduler = AirGuardScheduler()

