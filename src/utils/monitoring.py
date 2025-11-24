"""
Advanced monitoring and metrics collection for AirGuard.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import time
import asyncio
from collections import defaultdict, deque
import threading
from ..utils.logger import get_logger, performance_monitor

logger = get_logger("monitoring")


class MetricsCollector:
    """Collect and expose application metrics."""
    
    def __init__(self):
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        self._lock = threading.Lock()
        
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.counters[key] += value
            logger.debug(f"Counter {name} incremented by {value}", name=name, value=value, labels=labels)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.gauges[key] = value
            logger.debug(f"Gauge {name} set to {value}", name=name, value=value, labels=labels)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram metric."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.histograms[key].append(value)
            # Keep only last 1000 observations to prevent memory issues
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            logger.debug(f"Histogram {name} observed value {value}", name=name, value=value, labels=labels)
    
    def start_timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> "TimerContext":
        """Start timing an operation."""
        return TimerContext(self, name, labels)
    
    def record_timer(self, name: str, duration: float, labels: Optional[Dict[str, str]] = None):
        """Record a timer measurement."""
        with self._lock:
            key = self._get_metric_key(name, labels)
            self.timers[key].append(duration)
            # Keep only last 1000 measurements to prevent memory issues
            if len(self.timers[key]) > 1000:
                self.timers[key] = self.timers[key][-1000:]
            logger.debug(f"Timer {name} recorded duration {duration}", name=name, duration=duration, labels=labels)
    
    def _get_metric_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Generate a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{name}{{{label_str}}}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        with self._lock:
            metrics = {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "histograms": {},
                "timers": {}
            }
            
            # Calculate histogram statistics
            for key, values in self.histograms.items():
                if values:
                    metrics["histograms"][key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            
            # Calculate timer statistics
            for key, values in self.timers.items():
                if values:
                    metrics["timers"][key] = {
                        "count": len(values),
                        "sum": sum(values),
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "p50": self._percentile(values, 50),
                        "p95": self._percentile(values, 95),
                        "p99": self._percentile(values, 99)
                    }
            
            return metrics
    
    def _percentile(self, values: list, percentile: float) -> float:
        """Calculate percentile of a list of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            logger.info("All metrics reset")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, labels: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.labels)
            if exc_type is not None:
                # Record error if exception occurred
                self.collector.increment_counter(f"{self.name}_errors", labels=self.labels)


# Global metrics collector instance
metrics_collector = MetricsCollector()


class SystemMonitor:
    """Monitor system resources and application health."""
    
    def __init__(self):
        self.metrics_collector = metrics_collector
        self.logger = get_logger("system_monitor")
        self._stop_event = asyncio.Event()
        
    async def start_monitoring(self):
        """Start periodic system monitoring."""
        self.logger.info("Starting system monitoring")
        while not self._stop_event.is_set():
            try:
                await self._collect_system_metrics()
                await self._collect_application_metrics()
                # Wait 30 seconds before next collection
                await asyncio.wait_for(
                    self._stop_event.wait(), 
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                # Normal timeout, continue monitoring
                continue
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.logger.info("Stopping system monitoring")
        self._stop_event.set()
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge("system_cpu_percent", cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics_collector.set_gauge("system_memory_percent", memory.percent)
            self.metrics_collector.set_gauge("system_memory_available_bytes", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics_collector.set_gauge("system_disk_percent", (disk.used / disk.total) * 100)
            self.metrics_collector.set_gauge("system_disk_free_bytes", disk.free)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.metrics_collector.set_gauge("system_network_bytes_sent", net_io.bytes_sent)
            self.metrics_collector.set_gauge("system_network_bytes_recv", net_io.bytes_recv)
            
            self.logger.debug("System metrics collected", cpu=cpu_percent, memory=memory.percent)
            
        except ImportError:
            self.logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    async def _collect_application_metrics(self):
        """Collect application-level metrics."""
        try:
            # Collect performance monitor metrics
            perf_metrics = performance_monitor.get_metrics()
            
            # API metrics
            for endpoint, metrics in perf_metrics.items():
                if "call_count" in metrics:
                    self.metrics_collector.set_gauge(
                        "api_requests_total", 
                        metrics["call_count"],
                        {"endpoint": endpoint}
                    )
                    self.metrics_collector.set_gauge(
                        "api_avg_response_time_seconds", 
                        metrics["avg_duration"],
                        {"endpoint": endpoint}
                    )
                
                # Status code distribution
                for status_code, count in metrics.get("status_codes", {}).items():
                    self.metrics_collector.set_gauge(
                        "api_responses_total", 
                        count,
                        {"endpoint": endpoint, "status_code": str(status_code)}
                    )
            
            # Model metrics
            for model_name, metrics in perf_metrics.items():
                if "prediction_count" in metrics:
                    self.metrics_collector.set_gauge(
                        "model_predictions_total", 
                        metrics["prediction_count"],
                        {"model": model_name}
                    )
                    self.metrics_collector.set_gauge(
                        "model_avg_prediction_time_seconds", 
                        metrics["avg_duration"],
                        {"model": model_name}
                    )
                    
                    # Accuracy metrics if available
                    if metrics.get("accuracies"):
                        avg_accuracy = sum(metrics["accuracies"]) / len(metrics["accuracies"])
                        self.metrics_collector.set_gauge(
                            "model_avg_accuracy", 
                            avg_accuracy,
                            {"model": model_name}
                        )
            
            self.logger.debug("Application metrics collected")
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")


# Global system monitor instance
system_monitor = SystemMonitor()


def get_prometheus_metrics() -> str:
    """
    Generate Prometheus-formatted metrics.
    
    Returns:
        str: Prometheus metrics in text format
    """
    try:
        metrics = metrics_collector.get_metrics()
        lines = []
        
        # Add counters
        for key, value in metrics["counters"].items():
            name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        
        # Add gauges
        for key, value in metrics["gauges"].items():
            name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        
        # Add histograms
        for key, hist_data in metrics["histograms"].items():
            name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_count {hist_data['count']}")
            lines.append(f"{name}_sum {hist_data['sum']}")
            lines.append(f"{name} {{quantile=\"0.5\"}} {hist_data['p50']}")
            lines.append(f"{name} {{quantile=\"0.95\"}} {hist_data['p95']}")
            lines.append(f"{name} {{quantile=\"0.99\"}} {hist_data['p99']}")
        
        # Add timers
        for key, timer_data in metrics["timers"].items():
            name = key.replace("{", "_").replace("}", "").replace(",", "_").replace("=", "_")
            lines.append(f"# TYPE {name} summary")
            lines.append(f"{name}_count {timer_data['count']}")
            lines.append(f"{name}_sum {timer_data['sum']}")
            lines.append(f"{name} {{quantile=\"0.5\"}} {timer_data['p50']}")
            lines.append(f"{name} {{quantile=\"0.95\"}} {timer_data['p95']}")
            lines.append(f"{name} {{quantile=\"0.99\"}} {timer_data['p99']}")
        
        lines.append("")  # Empty line at end
        return "\n".join(lines)
        
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        return "# Error generating metrics\n"


# Decorators for easy metric collection
def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to count function calls.
    
    Args:
        metric_name: Name of the counter metric
        labels: Optional labels for the metric
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            metrics_collector.increment_counter(metric_name, labels=labels)
            return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            metrics_collector.increment_counter(metric_name, labels=labels)
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def time_execution(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator to time function execution.
    
    Args:
        metric_name: Name of the timer metric
        labels: Optional labels for the metric
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with metrics_collector.start_timer(metric_name, labels=labels):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with metrics_collector.start_timer(metric_name, labels=labels):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator