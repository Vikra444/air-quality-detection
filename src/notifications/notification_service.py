"""
Notification service for AirGuard alerts and advisories.
"""

import asyncio
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
from twilio.rest import Client
import firebase_admin
from firebase_admin import messaging
from ..config.settings import settings
from ..utils.logger import get_logger, performance_monitor
from ..utils.monitoring import time_execution, count_calls, metrics_collector
import time

logger = get_logger("notifications.service")


class NotificationService:
    """Service for sending notifications through multiple channels."""
    
    def __init__(self):
        self.email_enabled = settings.email_enabled
        self.sms_enabled = settings.sms_enabled
        self.push_enabled = settings.push_enabled
        
        # Initialize services
        self._init_email_service()
        self._init_sms_service()
        self._init_push_service()
    
    def _init_email_service(self):
        """Initialize email service."""
        if self.email_enabled and settings.smtp_host and settings.smtp_user:
            try:
                self.smtp_server = smtplib.SMTP(settings.smtp_host, settings.smtp_port)
                self.smtp_server.starttls()
                self.smtp_server.login(settings.smtp_user, settings.smtp_password)
                logger.info("Email service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize email service: {e}")
                self.email_enabled = False
        else:
            self.email_enabled = False
    
    def _init_sms_service(self):
        """Initialize SMS service."""
        if self.sms_enabled and settings.twilio_account_sid and settings.twilio_auth_token:
            try:
                self.twilio_client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
                logger.info("SMS service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize SMS service: {e}")
                self.sms_enabled = False
        else:
            self.sms_enabled = False
    
    def _init_push_service(self):
        """Initialize push notification service."""
        if self.push_enabled and settings.firebase_credentials_path:
            try:
                if not firebase_admin._apps:
                    firebase_admin.initialize_app()
                logger.info("Push notification service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize push service: {e}")
                self.push_enabled = False
        else:
            self.push_enabled = False
    
    @time_execution("alert_sending")
    async def send_alert(
        self,
        recipients: List[str],
        alert_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ):
        """Send alert through all enabled channels."""
        start_time = time.time()
        
        tasks = []
        
        # Send email alerts
        if self.email_enabled:
            tasks.append(self._send_email_alert(recipients, alert_type, message, data))
        
        # Send SMS alerts
        if self.sms_enabled:
            tasks.append(self._send_sms_alert(recipients, alert_type, message, data))
        
        # Send push notifications
        if self.push_enabled:
            tasks.append(self._send_push_alert(recipients, alert_type, message, data, priority))
        
        # Send webhook notifications
        tasks.append(self._send_webhook_alert(recipients, alert_type, message, data))
        
        # Execute all tasks concurrently
        successful_channels = 0
        if tasks:
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful_channels = len([r for r in results if not isinstance(r, Exception)])
                logger.info(f"Alert sent through {successful_channels}/{len(tasks)} channels")
            except Exception as e:
                logger.error(f"Error sending alerts: {e}")
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("AlertSending", duration)
        metrics_collector.increment_counter("alerts_sent", labels={"alert_type": alert_type, "priority": priority})
        metrics_collector.observe_histogram("alert_delivery_channels", successful_channels)
    
    @time_execution("email_alert")
    async def _send_email_alert(
        self,
        recipients: List[str],
        alert_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send email alert."""
        start_time = time.time()
        email_count = 0
        
        try:
            for recipient in recipients:
                if "@" not in recipient:
                    continue  # Skip non-email recipients
                
                msg = MIMEMultipart()
                msg['From'] = settings.email_from or settings.smtp_user
                msg['To'] = recipient
                msg['Subject'] = f"AirGuard Alert: {alert_type}"
                
                body = f"""
                AirGuard Alert - {alert_type}
                
                Message: {message}
                
                Time: {datetime.now().isoformat()}
                
                {f'Details: {data}' if data else ''}
                
                ---
                This is an automated alert from AirGuard system.
                """
                
                msg.attach(MIMEText(body, 'plain'))
                
                self.smtp_server.send_message(msg)
                email_count += 1
            
            logger.info(f"Email alerts sent to {email_count} recipients")
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("EmailAlert", duration)
            metrics_collector.increment_counter("email_alert_errors")
            raise e
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("EmailAlert", duration)
        metrics_collector.increment_counter("email_alerts_sent", labels={"alert_type": alert_type})
        metrics_collector.observe_histogram("emails_per_alert", email_count)
    
    @time_execution("sms_alert")
    async def _send_sms_alert(
        self,
        recipients: List[str],
        alert_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send SMS alert."""
        start_time = time.time()
        sms_count = 0
        
        try:
            for recipient in recipients:
                if not recipient.startswith('+'):
                    continue  # Skip non-phone recipients
                
                sms_body = f"AirGuard Alert: {alert_type}\n{message}"
                if data and 'aqi' in data:
                    sms_body += f"\nAQI: {data['aqi']}"
                
                self.twilio_client.messages.create(
                    body=sms_body,
                    from_=settings.twilio_phone_number,
                    to=recipient
                )
                sms_count += 1
            
            logger.info(f"SMS alerts sent to {sms_count} recipients")
        except Exception as e:
            logger.error(f"Error sending SMS alert: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("SMSAlert", duration)
            metrics_collector.increment_counter("sms_alert_errors")
            raise e
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("SMSAlert", duration)
        metrics_collector.increment_counter("sms_alerts_sent", labels={"alert_type": alert_type})
        metrics_collector.observe_histogram("sms_per_alert", sms_count)
    
    @time_execution("push_alert")
    async def _send_push_alert(
        self,
        recipients: List[str],
        alert_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        priority: str = "normal"
    ):
        """Send push notification."""
        start_time = time.time()
        
        try:
            # For demo purposes, we'll send to a topic
            # In production, you would send to specific device tokens
            notification = messaging.Notification(
                title=f"AirGuard Alert: {alert_type}",
                body=message
            )
            
            android_config = messaging.AndroidConfig(
                priority='high' if priority == 'high' else 'normal',
                notification=messaging.AndroidNotification(
                    icon='stock_ticker_update',
                    color='#f45342'
                )
            )
            
            apns_config = messaging.APNSConfig(
                payload=messaging.APNSPayload(
                    aps=messaging.Aps(badge=1, sound="default")
                )
            )
            
            message_obj = messaging.Message(
                notification=notification,
                data=data or {},
                topic="airguard_alerts",
                android=android_config,
                apns=apns_config
            )
            
            response = messaging.send(message_obj)
            logger.info(f"Push notification sent: {response}")
        except Exception as e:
            logger.error(f"Error sending push alert: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("PushAlert", duration)
            metrics_collector.increment_counter("push_alert_errors")
            raise e
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("PushAlert", duration)
        metrics_collector.increment_counter("push_alerts_sent", labels={"alert_type": alert_type, "priority": priority})
    
    @time_execution("webhook_alert")
    async def _send_webhook_alert(
        self,
        recipients: List[str],
        alert_type: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ):
        """Send webhook alert."""
        start_time = time.time()
        webhook_count = 0
        
        try:
            payload = {
                "alert_type": alert_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "data": data or {}
            }
            
            for recipient in recipients:
                if recipient.startswith('http'):
                    try:
                        response = requests.post(
                            recipient,
                            json=payload,
                            timeout=10
                        )
                        if response.status_code != 200:
                            logger.warning(f"Webhook failed for {recipient}: {response.status_code}")
                        else:
                            webhook_count += 1
                    except Exception as e:
                        logger.error(f"Webhook error for {recipient}: {e}")
            
            logger.info(f"Webhook alerts sent to {webhook_count} endpoints")
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("WebhookAlert", duration)
            metrics_collector.increment_counter("webhook_alert_errors")
            raise e
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("WebhookAlert", duration)
        metrics_collector.increment_counter("webhook_alerts_sent", labels={"alert_type": alert_type})
        metrics_collector.observe_histogram("webhooks_per_alert", webhook_count)
    
    def close(self):
        """Close notification services."""
        try:
            if self.email_enabled and hasattr(self, 'smtp_server'):
                self.smtp_server.quit()
        except Exception as e:
            logger.error(f"Error closing email service: {e}")


# Global notification service instance
notification_service = NotificationService()


class AlertManager:
    """Manage air quality alerts and notifications."""
    
    def __init__(self):
        self.notification_service = notification_service
        self.alert_thresholds = {
            "good": 50,
            "moderate": 100,
            "unhealthy_sensitive": 150,
            "unhealthy": 200,
            "very_unhealthy": 300,
            "hazardous": 500
        }
    
    @time_execution("alert_checking")
    async def check_and_send_alerts(
        self,
        location_id: str,
        aqi: float,
        recipients: List[str],
        previous_aqi: Optional[float] = None
    ):
        """Check air quality and send alerts if thresholds are exceeded."""
        start_time = time.time()
        alerts_sent = 0
        
        try:
            # Determine current alert level
            current_level = self._get_alert_level(aqi)
            
            # Check for significant changes
            significant_change = False
            change_level = "stable"
            if previous_aqi is not None:
                change = aqi - previous_aqi
                if abs(change) >= 20:  # 20 AQI point change is significant
                    significant_change = True
                    change_level = "improving" if change < 0 else "deteriorating"
            
            # Prepare alert data
            alert_data = {
                "location_id": location_id,
                "aqi": aqi,
                "alert_level": current_level,
                "previous_aqi": previous_aqi,
                "change": change_level if previous_aqi is not None else None
            }
            
            # Send alerts based on conditions
            alerts_sent_list = []
            
            # Critical alerts for hazardous conditions
            if aqi >= self.alert_thresholds["hazardous"]:
                await self.notification_service.send_alert(
                    recipients,
                    "HAZARDOUS_CONDITIONS",
                    f"Air quality is hazardous (AQI: {aqi}). Avoid all outdoor activities.",
                    alert_data,
                    priority="high"
                )
                alerts_sent_list.append("hazardous")
                alerts_sent += 1
            
            # Very unhealthy alerts
            elif aqi >= self.alert_thresholds["very_unhealthy"]:
                await self.notification_service.send_alert(
                    recipients,
                    "VERY_UNHEALTHY_CONDITIONS",
                    f"Air quality is very unhealthy (AQI: {aqi}). Avoid outdoor activities.",
                    alert_data,
                    priority="high"
                )
                alerts_sent_list.append("very_unhealthy")
                alerts_sent += 1
            
            # Unhealthy alerts
            elif aqi >= self.alert_thresholds["unhealthy"]:
                await self.notification_service.send_alert(
                    recipients,
                    "UNHEALTHY_CONDITIONS",
                    f"Air quality is unhealthy (AQI: {aqi}). Limit outdoor activities.",
                    alert_data,
                    priority="normal"
                )
                alerts_sent_list.append("unhealthy")
                alerts_sent += 1
            
            # Significant change alerts
            if significant_change and change_level == "deteriorating":
                await self.notification_service.send_alert(
                    recipients,
                    "AIR_QUALITY_DETERIORATING",
                    f"Air quality is deteriorating significantly (AQI: {aqi}, change: +{aqi - previous_aqi:.1f})",
                    alert_data,
                    priority="normal"
                )
                alerts_sent_list.append("deteriorating")
                alerts_sent += 1
            elif significant_change and change_level == "improving":
                await self.notification_service.send_alert(
                    recipients,
                    "AIR_QUALITY_IMPROVING",
                    f"Air quality is improving (AQI: {aqi}, change: {aqi - previous_aqi:.1f})",
                    alert_data,
                    priority="normal"
                )
                alerts_sent_list.append("improving")
                alerts_sent += 1
            
            if alerts_sent_list:
                logger.info(f"Sent {len(alerts_sent_list)} alerts for {location_id}: {', '.join(alerts_sent_list)}")
            
            # Record metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("AlertChecking", duration)
            metrics_collector.increment_counter("alert_checks")
            metrics_collector.observe_histogram("alerts_per_check", alerts_sent)
            
        except Exception as e:
            logger.error(f"Error checking and sending alerts: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("AlertChecking", duration)
            metrics_collector.increment_counter("alert_check_errors")
            raise e
    
    def _get_alert_level(self, aqi: float) -> str:
        """Get alert level based on AQI value."""
        if aqi <= self.alert_thresholds["good"]:
            return "good"
        elif aqi <= self.alert_thresholds["moderate"]:
            return "moderate"
        elif aqi <= self.alert_thresholds["unhealthy_sensitive"]:
            return "unhealthy_sensitive"
        elif aqi <= self.alert_thresholds["unhealthy"]:
            return "unhealthy"
        elif aqi <= self.alert_thresholds["very_unhealthy"]:
            return "very_unhealthy"
        else:
            return "hazardous"
    
    @time_execution("health_advisory")
    async def send_health_advisory(
        self,
        location_id: str,
        aqi: float,
        vulnerable_groups: List[str],
        recipients: List[str]
    ):
        """Send health advisory for vulnerable groups."""
        start_time = time.time()
        
        try:
            message = f"Health advisory for {location_id} (AQI: {aqi}). "
            
            if aqi >= self.alert_thresholds["unhealthy"]:
                message += "Unhealthy conditions. Sensitive groups should avoid outdoor activities."
            elif aqi >= self.alert_thresholds["unhealthy_sensitive"]:
                message += "Unhealthy for sensitive groups. Take precautions."
            
            alert_data = {
                "location_id": location_id,
                "aqi": aqi,
                "vulnerable_groups": vulnerable_groups
            }
            
            await self.notification_service.send_alert(
                recipients,
                "HEALTH_ADVISORY",
                message,
                alert_data,
                priority="normal"
            )
            
            logger.info(f"Health advisory sent for {location_id} to {len(recipients)} recipients")
            
            # Record metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("HealthAdvisory", duration)
            metrics_collector.increment_counter("health_advisories_sent")
            
        except Exception as e:
            logger.error(f"Error sending health advisory: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("HealthAdvisory", duration)
            metrics_collector.increment_counter("health_advisory_errors")
            raise e


# Global alert manager instance
alert_manager = AlertManager()