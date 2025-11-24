"""
Data quality assurance module for AirGuard.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from .models import AirQualityData
from ..utils.logger import get_logger, performance_monitor
from ..utils.monitoring import time_execution, count_calls, metrics_collector
import time

logger = get_logger("data.quality")


class DataQualityAssurance:
    """Ensure data quality and detect anomalies in air quality data."""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.quality_thresholds = {
            "pm25": {"min": 0, "max": 1000},
            "pm10": {"min": 0, "max": 1500},
            "no2": {"min": 0, "max": 500},
            "co": {"min": 0, "max": 50},
            "o3": {"min": 0, "max": 500},
            "so2": {"min": 0, "max": 1000},
            "aqi": {"min": 0, "max": 500}
        }
    
    @time_execution("data_validation")
    def validate_data(self, data: AirQualityData) -> Tuple[bool, List[str]]:
        """Validate air quality data and return validation status and issues."""
        start_time = time.time()
        
        issues = []
        
        # Check required fields
        required_fields = ["timestamp", "location_id", "latitude", "longitude", "aqi"]
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                issues.append(f"Missing required field: {field}")
        
        # Validate ranges
        for pollutant, thresholds in self.quality_thresholds.items():
            if hasattr(data, pollutant):
                value = getattr(data, pollutant)
                if value is not None:
                    if value < thresholds["min"] or value > thresholds["max"]:
                        issues.append(f"{pollutant} value {value} out of range [{thresholds['min']}, {thresholds['max']}]")
        
        # Validate coordinates
        if data.latitude < -90 or data.latitude > 90:
            issues.append(f"Latitude {data.latitude} out of range [-90, 90]")
        
        if data.longitude < -180 or data.longitude > 180:
            issues.append(f"Longitude {data.longitude} out of range [-180, 180]")
        
        # Validate timestamp
        if data.timestamp > datetime.now() + timedelta(hours=1):
            issues.append("Timestamp is in the future")
        
        is_valid = len(issues) == 0
        if not is_valid:
            logger.warning(f"Data validation failed for {data.location_id}: {issues}")
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("DataValidation", duration)
        metrics_collector.increment_counter("data_validations", labels={"valid": str(is_valid)})
        
        return is_valid, issues
    
    @time_execution("anomaly_detection")
    def detect_anomalies(self, data_list: List[AirQualityData]) -> List[Dict[str, Any]]:
        """Detect anomalies in a list of air quality data points."""
        start_time = time.time()
        
        if len(data_list) < 10:
            return []  # Need sufficient data for anomaly detection
        
        try:
            # Convert to DataFrame
            df_data = []
            for data in data_list:
                df_data.append({
                    "pm25": data.pm25,
                    "pm10": data.pm10,
                    "no2": data.no2,
                    "co": data.co,
                    "o3": data.o3,
                    "so2": data.so2,
                    "aqi": data.aqi,
                    "temperature": data.temperature or 20,
                    "humidity": data.humidity or 50,
                    "wind_speed": data.wind_speed or 3
                })
            
            df = pd.DataFrame(df_data)
            
            # Standardize features
            features = ["pm25", "pm10", "no2", "co", "o3", "so2", "aqi", "temperature", "humidity", "wind_speed"]
            X = self.scaler.fit_transform(df[features])
            
            # Detect anomalies
            anomaly_labels = self.isolation_forest.fit_predict(X)
            
            # Identify anomalous data points
            anomalies = []
            for i, is_anomaly in enumerate(anomaly_labels):
                if is_anomaly == -1:  # -1 indicates anomaly
                    anomalies.append({
                        "index": i,
                        "data_point": data_list[i],
                        "anomaly_score": self.isolation_forest.decision_function(X)[i],
                        "reason": "Statistical anomaly detected"
                    })
            
            if anomalies:
                logger.info(f"Detected {len(anomalies)} anomalies in batch of {len(data_list)} data points")
            
            # Record metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("AnomalyDetection", duration)
            metrics_collector.increment_counter("anomalies_detected", labels={"count": str(len(anomalies))})
            
            return anomalies
        
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("AnomalyDetection", duration)
            metrics_collector.increment_counter("anomaly_detection_errors")
            return []
    
    @time_execution("quality_scoring")
    def calculate_data_quality_score(self, data: AirQualityData) -> float:
        """Calculate a quality score for air quality data (0-1)."""
        start_time = time.time()
        
        score = 1.0
        
        # Check for missing values
        required_fields = ["pm25", "pm10", "aqi"]
        missing_count = 0
        for field in required_fields:
            if not hasattr(data, field) or getattr(data, field) is None:
                missing_count += 1
        
        score -= (missing_count / len(required_fields)) * 0.3
        
        # Check data freshness (penalize old data)
        age_hours = (datetime.now() - data.timestamp).total_seconds() / 3600
        if age_hours > 24:
            score -= 0.2
        elif age_hours > 12:
            score -= 0.1
        
        # Check for extreme values
        extreme_count = 0
        for pollutant, thresholds in self.quality_thresholds.items():
            if hasattr(data, pollutant):
                value = getattr(data, pollutant)
                if value is not None:
                    if value > thresholds["max"] * 0.8:  # Within 20% of max
                        extreme_count += 1
        
        score -= (extreme_count / len(self.quality_thresholds)) * 0.2
        
        # Bonus for complete data
        optional_fields = ["no2", "co", "o3", "so2", "temperature", "humidity", "wind_speed"]
        present_count = 0
        for field in optional_fields:
            if hasattr(data, field) and getattr(data, field) is not None:
                present_count += 1
        
        score += (present_count / len(optional_fields)) * 0.2
        
        final_score = max(0.0, min(1.0, score))
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("QualityScoring", duration)
        metrics_collector.observe_histogram("data_quality_scores", final_score)
        
        return final_score
    
    @time_execution("data_cleaning")
    def clean_data(self, data: AirQualityData) -> AirQualityData:
        """Clean and normalize air quality data."""
        start_time = time.time()
        
        # Ensure all required fields have values
        if data.pm25 is None:
            data.pm25 = 0.0
        if data.pm10 is None:
            data.pm10 = 0.0
        if data.aqi is None:
            data.aqi = 0.0
        
        # Clamp values to valid ranges
        for pollutant, thresholds in self.quality_thresholds.items():
            if hasattr(data, pollutant):
                value = getattr(data, pollutant)
                if value is not None:
                    setattr(data, pollutant, max(thresholds["min"], min(thresholds["max"], value)))
        
        # Ensure coordinates are valid
        data.latitude = max(-90, min(90, data.latitude))
        data.longitude = max(-180, min(180, data.longitude))
        
        # Ensure humidity is valid
        if data.humidity is not None:
            data.humidity = max(0, min(100, data.humidity))
        
        # Ensure wind direction is valid
        if data.wind_direction is not None:
            data.wind_direction = data.wind_direction % 360
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("DataCleaning", duration)
        metrics_collector.increment_counter("data_cleaning_operations")
        
        return data
    
    @time_execution("sensor_drift_detection")
    def detect_sensor_drift(self, historical_data: List[AirQualityData], window_days: int = 7) -> Dict[str, Any]:
        """Detect sensor drift by comparing recent data with historical patterns."""
        start_time = time.time()
        
        if len(historical_data) < window_days * 24:  # Need at least window_days of hourly data
            return {"status": "insufficient_data", "drift_detected": False}
        
        try:
            # Get recent data (last window_days)
            recent_data = historical_data[-(window_days * 24):]
            # Get baseline data (before recent data)
            baseline_data = historical_data[-(window_days * 48):-(window_days * 24)]
            
            if len(baseline_data) < window_days * 24:
                return {"status": "insufficient_baseline", "drift_detected": False}
            
            # Calculate statistics for both periods
            recent_stats = {}
            baseline_stats = {}
            
            pollutants = ["pm25", "pm10", "no2", "co", "o3", "so2", "aqi"]
            
            for pollutant in pollutants:
                recent_values = [getattr(d, pollutant) for d in recent_data if getattr(d, pollutant) is not None]
                baseline_values = [getattr(d, pollutant) for d in baseline_data if getattr(d, pollutant) is not None]
                
                if recent_values and baseline_values:
                    recent_stats[pollutant] = {
                        "mean": np.mean(recent_values),
                        "std": np.std(recent_values)
                    }
                    baseline_stats[pollutant] = {
                        "mean": np.mean(baseline_values),
                        "std": np.std(baseline_values)
                    }
            
            # Detect drift using t-test
            drift_detected = False
            drift_details = {}
            
            for pollutant in recent_stats:
                if pollutant in baseline_stats:
                    # Perform t-test
                    recent_vals = [getattr(d, pollutant) for d in recent_data if getattr(d, pollutant) is not None]
                    baseline_vals = [getattr(d, pollutant) for d in baseline_data if getattr(d, pollutant) is not None]
                    
                    if len(recent_vals) > 1 and len(baseline_vals) > 1:
                        t_stat, p_value = stats.ttest_ind(recent_vals, baseline_vals)
                        
                        # Consider drift significant if p < 0.05 and mean difference > 10%
                        mean_diff = abs(recent_stats[pollutant]["mean"] - baseline_stats[pollutant]["mean"])
                        mean_ratio = mean_diff / max(baseline_stats[pollutant]["mean"], 1)
                        
                        if p_value < 0.05 and mean_ratio > 0.1:
                            drift_detected = True
                            drift_details[pollutant] = {
                                "t_statistic": t_stat,
                                "p_value": p_value,
                                "mean_difference": mean_diff,
                                "mean_ratio": mean_ratio,
                                "recent_mean": recent_stats[pollutant]["mean"],
                                "baseline_mean": baseline_stats[pollutant]["mean"]
                            }
            
            result = {
                "status": "success",
                "drift_detected": drift_detected,
                "details": drift_details,
                "recent_period": f"Last {window_days} days",
                "baseline_period": f"{window_days*2} to {window_days} days ago"
            }
            
            # Record metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("SensorDriftDetection", duration)
            metrics_collector.increment_counter("sensor_drift_checks", labels={"drift_detected": str(drift_detected)})
            
            return result
        
        except Exception as e:
            logger.error(f"Error in sensor drift detection: {e}")
            # Record error metrics
            duration = time.time() - start_time
            performance_monitor.record_model_prediction("SensorDriftDetection", duration)
            metrics_collector.increment_counter("sensor_drift_errors")
            
            return {"status": "error", "error": str(e), "drift_detected": False}
    
    @time_execution("quality_report_generation")
    def generate_quality_report(self, data_list: List[AirQualityData]) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        start_time = time.time()
        
        if not data_list:
            return {"status": "no_data", "report": {}}
        
        total_points = len(data_list)
        valid_points = 0
        anomaly_count = 0
        avg_quality_score = 0.0
        
        # Validate all data points
        validation_issues = []
        quality_scores = []
        
        for data in data_list:
            is_valid, issues = self.validate_data(data)
            if is_valid:
                valid_points += 1
            else:
                validation_issues.extend(issues)
            
            quality_score = self.calculate_data_quality_score(data)
            quality_scores.append(quality_score)
        
        avg_quality_score = np.mean(quality_scores) if quality_scores else 0.0
        
        # Detect anomalies
        anomalies = self.detect_anomalies(data_list)
        anomaly_count = len(anomalies)
        
        # Calculate completeness
        completeness = valid_points / total_points if total_points > 0 else 0.0
        
        report = {
            "total_data_points": total_points,
            "valid_data_points": valid_points,
            "completeness": completeness,
            "anomalies_detected": anomaly_count,
            "anomaly_rate": anomaly_count / total_points if total_points > 0 else 0.0,
            "average_quality_score": avg_quality_score,
            "validation_issues": validation_issues[:10],  # Top 10 issues
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Data quality report generated: {valid_points}/{total_points} valid, {anomaly_count} anomalies")
        
        # Record metrics
        duration = time.time() - start_time
        performance_monitor.record_model_prediction("QualityReportGeneration", duration)
        metrics_collector.increment_counter("quality_reports_generated")
        metrics_collector.observe_histogram("data_completeness", completeness)
        metrics_collector.observe_histogram("anomaly_rates", anomaly_count / total_points if total_points > 0 else 0.0)
        
        return {
            "status": "success",
            "report": report
        }


# Global quality assurance instance
quality_assurance = DataQualityAssurance()