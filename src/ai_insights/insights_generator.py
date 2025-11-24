"""
AI Insights generator combining trend analysis, anomaly detection, and source attribution.
"""

from typing import Dict, Any, List, Optional

from .trend_analyzer import TrendAnalyzer
from .anomaly_detector import AnomalyDetector
from .source_attribution import SourceAttributionEngine
from ..utils.logger import get_logger

logger = get_logger("ai_insights.insights_generator")


class InsightsGenerator:
    """Generate comprehensive AI insights for air quality."""
    
    def __init__(self):
        """Initialize insights generator."""
        self.trend_analyzer = TrendAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.source_attribution = SourceAttributionEngine()
    
    def generate_insights(
        self,
        current_data: Dict[str, Any],
        historical_data: Optional[List[Dict[str, Any]]] = None,
        weather_data: Optional[Dict[str, Any]] = None,
        previous_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive AI insights.
        
        Args:
            current_data: Current air quality data
            historical_data: Historical data for trend/anomaly analysis
            weather_data: Current weather data
            previous_data: Previous air quality data for change analysis
        
        Returns:
            Dictionary with comprehensive insights
        """
        try:
            insights = {
                "timestamp": current_data.get("timestamp"),
                "current_aqi": current_data.get("aqi", 0)
            }
            
            # Trend analysis
            if historical_data and len(historical_data) > 1:
                insights["trend"] = self.trend_analyzer.analyze_trend(historical_data, current_data)
            
            # Anomaly detection
            if historical_data:
                insights["anomaly"] = self.anomaly_detector.detect_anomaly(
                    current_data,
                    historical_data
                )
            
            # Source attribution
            insights["source_attribution"] = self.source_attribution.analyze(
                current_data,
                weather_data,
                previous_data
            )
            
            # Generate summary insight
            insights["summary"] = self._generate_summary_insight(insights)
            
            return insights
        
        except Exception as e:
            logger.error(f"Error generating insights: {e}", error=str(e))
            return {"error": str(e)}
    
    def _generate_summary_insight(self, insights: Dict[str, Any]) -> str:
        """Generate summary insight text."""
        summary_parts = []
        
        # Add trend insight
        if "trend" in insights:
            trend = insights["trend"]
            if trend.get("direction") != "unknown":
                summary_parts.append(trend.get("explanation", ""))
        
        # Add anomaly insight
        if "anomaly" in insights:
            anomaly = insights["anomaly"]
            if anomaly.get("is_anomaly"):
                summary_parts.append(anomaly.get("explanation", ""))
        
        # Add source attribution
        if "source_attribution" in insights:
            source_attr = insights["source_attribution"]
            if "summary" in source_attr:
                summary_parts.append(source_attr["summary"])
        
        # Combine into single insight
        if summary_parts:
            return " ".join(summary_parts)
        else:
            return "Air quality monitoring active. No significant insights at this time."

