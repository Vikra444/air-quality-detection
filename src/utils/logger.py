"""
Enhanced logging utilities for AirGuard.
"""

import logging
import sys
from typing import Optional, Dict, Any
from datetime import datetime
import json
import time
from functools import wraps
from ..config.settings import settings

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Create formatter
formatter = logging.Formatter(LOG_FORMAT)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# Create file handler
file_handler = logging.FileHandler("airguard.log")
file_handler.setFormatter(formatter)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, settings.log_level.upper()))
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Log to file with rotation
try:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        "airguard.log", 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
except ImportError:
    pass  # Fall back to regular file handler


class StructuredLogger:
    """Structured logger that outputs JSON format for better monitoring."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    def _log(self, level: str, message: str, **kwargs):
        """Log message with structured data."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }
        
        # Output as JSON for structured logging
        if settings.debug:
            # In debug mode, output human-readable format
            log_message = f"{log_data['timestamp']} - {log_data['level']} - {log_data['message']}"
            if len(kwargs) > 0:
                log_message += f" - {kwargs}"
            getattr(self.logger, level.lower())(log_message)
        else:
            # In production, output JSON
            getattr(self.logger, level.lower())(json.dumps(log_data, default=str))
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)
    
    def success(self, message: str, **kwargs):
        """Log success message."""
        self._log("INFO", message, status="success", **kwargs)


def get_logger(name: str) -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger(name)


class PerformanceMonitor:
    """Monitor performance metrics and system health."""
    
    def __init__(self):
        self.metrics = {}
        self.logger = get_logger("performance")
    
    def record_api_call(self, endpoint: str, duration: float, status_code: int):
        """Record API call metrics."""
        if endpoint not in self.metrics:
            self.metrics[endpoint] = {
                "call_count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "status_codes": {}
            }
        
        endpoint_metrics = self.metrics[endpoint]
        endpoint_metrics["call_count"] += 1
        endpoint_metrics["total_duration"] += duration
        
        # Update average duration
        endpoint_metrics["avg_duration"] = (
            endpoint_metrics["total_duration"] / endpoint_metrics["call_count"]
        )
        
        # Track status codes
        if status_code not in endpoint_metrics["status_codes"]:
            endpoint_metrics["status_codes"][status_code] = 0
        endpoint_metrics["status_codes"][status_code] += 1
        
        # Log slow requests
        if duration > 5.0:  # Log requests taking more than 5 seconds
            self.logger.warning(
                f"Slow API call: {endpoint} took {duration:.2f}s",
                endpoint=endpoint,
                duration=duration,
                status_code=status_code
            )
    
    def record_model_prediction(self, model_name: str, duration: float, accuracy: Optional[float] = None):
        """Record model prediction metrics."""
        if model_name not in self.metrics:
            self.metrics[model_name] = {
                "prediction_count": 0,
                "total_duration": 0.0,
                "avg_duration": 0.0,
                "accuracies": []
            }
        
        model_metrics = self.metrics[model_name]
        model_metrics["prediction_count"] += 1
        model_metrics["total_duration"] += duration
        
        # Update average duration
        model_metrics["avg_duration"] = (
            model_metrics["total_duration"] / model_metrics["prediction_count"]
        )
        
        # Track accuracy if provided
        if accuracy is not None:
            model_metrics["accuracies"].append(accuracy)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return self.metrics
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = {}
    
    def log_system_metrics(self):
        """Log system-level metrics."""
        try:
            import psutil
            import GPUtil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('/')
            
            # Network stats
            net_io = psutil.net_io_counters()
            
            # GPU usage (if available)
            gpu_info = []
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info.append({
                        "id": gpu.id,
                        "name": gpu.name,
                        "load": gpu.load * 100,
                        "memory_util": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature
                    })
            except:
                pass  # GPU monitoring optional
            
            system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": (disk.used / disk.total) * 100,
                "disk_free_gb": disk.free / (1024 * 1024 * 1024),
                "network_bytes_sent": net_io.bytes_sent,
                "network_bytes_recv": net_io.bytes_recv,
                "gpu_info": gpu_info
            }
            
            self.logger.info("System metrics collected", **system_metrics)
            return system_metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return {}


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(metric_name: str):
    """
    Decorator to monitor function performance.
    
    Args:
        metric_name: Name to identify the metric in monitoring
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_model_prediction(metric_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_model_prediction(metric_name, duration)
                raise e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_model_prediction(metric_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_model_prediction(metric_name, duration)
                raise e
        
        # Return appropriate wrapper based on function type
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


import asyncio