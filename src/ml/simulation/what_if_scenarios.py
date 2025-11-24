"""
What-if scenario simulator for air quality policy analysis.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...utils.logger import get_logger

logger = get_logger("ml.simulation.what_if")


class WhatIfSimulator:
    """Simulate what-if scenarios for air quality improvement."""
    
    def __init__(self, predictor=None):
        """
        Initialize what-if simulator.
        
        Args:
            predictor: AirQualityPredictor instance for making predictions
        """
        self.predictor = predictor
    
    def simulate(
        self,
        current_data: Dict[str, Any],
        scenario: Dict[str, Any],
        prediction_horizon: int = 24
    ) -> Dict[str, Any]:
        """
        Simulate a what-if scenario.
        
        Args:
            current_data: Current air quality and weather data
            scenario: Scenario parameters (e.g., {"traffic_reduction": 0.2, "green_cover_increase": 0.1})
            prediction_horizon: Hours to predict ahead
        
        Returns:
            Dictionary with simulation results
        """
        try:
            baseline_aqi = current_data.get("aqi", 0)
            
            # Create modified data based on scenario
            modified_data = current_data.copy()
            
            # Apply scenario changes
            changes_applied = {}
            
            # Traffic reduction
            if "traffic_reduction" in scenario:
                reduction = scenario["traffic_reduction"]
                # Reduce NO2 and CO proportionally
                if "no2" in modified_data:
                    modified_data["no2"] = modified_data["no2"] * (1 - reduction)
                    changes_applied["no2"] = {
                        "original": current_data.get("no2", 0),
                        "modified": modified_data["no2"],
                        "change_percent": -reduction * 100
                    }
                if "co" in modified_data:
                    modified_data["co"] = modified_data["co"] * (1 - reduction)
                    changes_applied["co"] = {
                        "original": current_data.get("co", 0),
                        "modified": modified_data["co"],
                        "change_percent": -reduction * 100
                    }
            
            # Green cover increase
            if "green_cover_increase" in scenario:
                increase = scenario["green_cover_increase"]
                # Reduce PM2.5 and PM10 (plants filter air)
                if "pm25" in modified_data:
                    reduction_factor = increase * 0.15  # 15% reduction per 10% green increase
                    modified_data["pm25"] = modified_data["pm25"] * (1 - reduction_factor)
                    changes_applied["pm25"] = {
                        "original": current_data.get("pm25", 0),
                        "modified": modified_data["pm25"],
                        "change_percent": -reduction_factor * 100
                    }
                if "pm10" in modified_data:
                    reduction_factor = increase * 0.15
                    modified_data["pm10"] = modified_data["pm10"] * (1 - reduction_factor)
                    changes_applied["pm10"] = {
                        "original": current_data.get("pm10", 0),
                        "modified": modified_data["pm10"],
                        "change_percent": -reduction_factor * 100
                    }
            
            # Industrial emission reduction
            if "industrial_reduction" in scenario:
                reduction = scenario["industrial_reduction"]
                # Reduce SO2 and PM2.5
                if "so2" in modified_data:
                    modified_data["so2"] = modified_data["so2"] * (1 - reduction)
                    changes_applied["so2"] = {
                        "original": current_data.get("so2", 0),
                        "modified": modified_data["so2"],
                        "change_percent": -reduction * 100
                    }
                if "pm25" in modified_data:
                    modified_data["pm25"] = modified_data["pm25"] * (1 - reduction * 0.5)  # Partial impact
                    if "pm25" not in changes_applied:
                        changes_applied["pm25"] = {
                            "original": current_data.get("pm25", 0),
                            "modified": modified_data["pm25"],
                            "change_percent": -reduction * 50
                        }
            
            # Wind speed increase (weather intervention)
            if "wind_speed_increase" in scenario:
                increase = scenario["wind_speed_increase"]
                if "wind_speed" in modified_data:
                    modified_data["wind_speed"] = modified_data["wind_speed"] + increase
                    changes_applied["wind_speed"] = {
                        "original": current_data.get("wind_speed", 0),
                        "modified": modified_data["wind_speed"],
                        "change_percent": (increase / current_data.get("wind_speed", 1)) * 100
                    }
                # Higher wind disperses pollutants
                for pollutant in ["pm25", "pm10", "no2"]:
                    if pollutant in modified_data:
                        dispersal_factor = min(increase * 0.1, 0.3)  # Max 30% reduction
                        modified_data[pollutant] = modified_data[pollutant] * (1 - dispersal_factor)
            
            # Recalculate AQI based on modified pollutants
            modified_aqi = self._estimate_aqi_from_pollutants(modified_data)
            
            # Calculate impact
            aqi_change = modified_aqi - baseline_aqi
            change_percent = (aqi_change / baseline_aqi * 100) if baseline_aqi > 0 else 0
            
            # Generate predictions if predictor available
            predictions = None
            if self.predictor:
                try:
                    # Create prediction request
                    from ...data.models import PredictionRequest
                    pred_request = PredictionRequest(
                        location_id=current_data.get("location_id", "simulation"),
                        latitude=current_data.get("latitude", 0),
                        longitude=current_data.get("longitude", 0),
                        prediction_horizon=prediction_horizon
                    )
                    # Use modified data for prediction
                    pred_result = self.predictor.predict(pred_request, input_data=modified_data)
                    predictions = {
                        "predicted_aqi": pred_result.predictions.get("aqi", [modified_aqi])[0] if isinstance(pred_result.predictions, dict) else modified_aqi,
                        "confidence": pred_result.confidence_score
                    }
                except Exception as e:
                    logger.warning(f"Failed to generate predictions: {e}")
            
            # Generate explanation
            explanation = self._generate_explanation(
                scenario,
                baseline_aqi,
                modified_aqi,
                aqi_change,
                changes_applied
            )
            
            return {
                "scenario": scenario,
                "baseline_aqi": baseline_aqi,
                "simulated_aqi": modified_aqi,
                "aqi_change": aqi_change,
                "change_percent": change_percent,
                "changes_applied": changes_applied,
                "predictions": predictions,
                "explanation": explanation,
                "confidence": self._calculate_confidence(scenario, changes_applied)
            }
        
        except Exception as e:
            logger.error(f"Error in what-if simulation: {e}", error=str(e))
            return {"error": str(e)}
    
    def _estimate_aqi_from_pollutants(self, data: Dict[str, Any]) -> float:
        """Estimate AQI from pollutant concentrations."""
        # Simple AQI estimation based on highest pollutant
        pm25 = data.get("pm25", 0)
        pm10 = data.get("pm10", 0)
        no2 = data.get("no2", 0)
        o3 = data.get("o3", 0)
        co = data.get("co", 0)
        so2 = data.get("so2", 0)
        
        # Convert to AQI sub-indices (simplified)
        aqi_pm25 = min(pm25 * 2, 500)  # Rough conversion
        aqi_pm10 = min(pm10 * 1.5, 500)
        aqi_no2 = min(no2 * 2, 500)
        aqi_o3 = min(o3 * 1.2, 500)
        aqi_co = min(co * 10, 500)
        aqi_so2 = min(so2 * 1.5, 500)
        
        # AQI is the maximum of all sub-indices
        estimated_aqi = max(aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_co, aqi_so2)
        
        return estimated_aqi
    
    def _generate_explanation(
        self,
        scenario: Dict[str, Any],
        baseline_aqi: float,
        simulated_aqi: float,
        aqi_change: float,
        changes_applied: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of simulation results."""
        direction = "improve" if aqi_change < 0 else "worsen"
        magnitude = abs(aqi_change)
        
        explanation = f"Simulating scenario: "
        
        # List scenario parameters
        scenario_parts = []
        if "traffic_reduction" in scenario:
            scenario_parts.append(f"reduce traffic by {scenario['traffic_reduction']*100:.1f}%")
        if "green_cover_increase" in scenario:
            scenario_parts.append(f"increase green cover by {scenario['green_cover_increase']*100:.1f}%")
        if "industrial_reduction" in scenario:
            scenario_parts.append(f"reduce industrial emissions by {scenario['industrial_reduction']*100:.1f}%")
        if "wind_speed_increase" in scenario:
            scenario_parts.append(f"increase wind speed by {scenario['wind_speed_increase']:.1f} m/s")
        
        explanation += ", ".join(scenario_parts) + ". "
        
        explanation += f"Expected AQI would {direction} by {magnitude:.1f} points "
        explanation += f"(from {baseline_aqi:.1f} to {simulated_aqi:.1f}). "
        
        # Highlight key changes
        if changes_applied:
            top_change = max(
                changes_applied.items(),
                key=lambda x: abs(x[1].get("change_percent", 0)),
                default=None
            )
            if top_change:
                pollutant, change_data = top_change
                explanation += f"Primary impact: {pollutant.upper()} would change by {change_data['change_percent']:.1f}%. "
        
        return explanation
    
    def _calculate_confidence(
        self,
        scenario: Dict[str, Any],
        changes_applied: Dict[str, Any]
    ) -> float:
        """Calculate confidence in simulation results."""
        # Base confidence
        confidence = 0.7
        
        # Higher confidence if we have more changes
        if len(changes_applied) > 2:
            confidence += 0.1
        
        # Higher confidence for traffic/green cover (well-studied)
        if "traffic_reduction" in scenario or "green_cover_increase" in scenario:
            confidence += 0.1
        
        return min(confidence, 0.9)

