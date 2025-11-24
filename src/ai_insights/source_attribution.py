"""
Main source attribution engine combining classifier, causal analysis, and explainer.
"""

from typing import Dict, Any, Optional, List

from .source_attribution.source_classifier import SourceClassifier, PollutionSource
from .source_attribution.causal_analysis import CausalAnalyzer
from .source_attribution.attribution_explainer import AttributionExplainer
from ...utils.logger import get_logger

logger = get_logger("ai_insights.source_attribution")


class SourceAttributionEngine:
    """Main engine for source attribution analysis."""
    
    def __init__(self):
        """Initialize source attribution engine."""
        self.classifier = SourceClassifier()
        self.causal_analyzer = CausalAnalyzer()
        self.explainer = AttributionExplainer()
    
    def analyze(
        self,
        air_quality_data: Dict[str, Any],
        weather_data: Optional[Dict[str, Any]] = None,
        previous_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive source attribution analysis.
        
        Args:
            air_quality_data: Current air quality data
            weather_data: Current weather data
            previous_data: Previous air quality data for change analysis
        
        Returns:
            Dictionary with source attribution results
        """
        try:
            # Classify source
            source_result = self.classifier.classify(air_quality_data, weather_data)
            
            # Explain change if previous data available
            if previous_data:
                change_explanation = self.explainer.explain_aqi_change(
                    air_quality_data,
                    previous_data,
                    weather_data
                )
            else:
                change_explanation = self.explainer.explain_aqi_change(
                    air_quality_data,
                    None,
                    weather_data
                )
            
            return {
                "source_classification": source_result,
                "change_analysis": change_explanation,
                "summary": self._generate_summary(source_result, change_explanation)
            }
        
        except Exception as e:
            logger.error(f"Error in source attribution analysis: {e}", error=str(e))
            return {"error": str(e)}
    
    def _generate_summary(
        self,
        source_result: Dict[str, Any],
        change_analysis: Dict[str, Any]
    ) -> str:
        """Generate summary of source attribution."""
        primary_source = source_result.get("primary_source", "unknown")
        confidence = source_result.get("confidence", 0.0)
        
        summary = f"Primary pollution source: {primary_source} (confidence: {confidence*100:.1f}%). "
        
        if "explanation" in change_analysis:
            summary += change_analysis["explanation"]
        
        return summary

