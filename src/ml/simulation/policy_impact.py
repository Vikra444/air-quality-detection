"""
Policy impact analyzer for air quality policies.
"""

from typing import Dict, Any, List, Optional
from .what_if_scenarios import WhatIfSimulator

from ...utils.logger import get_logger

logger = get_logger("ml.simulation.policy_impact")


class PolicyImpactAnalyzer:
    """Analyze impact of air quality policies."""
    
    def __init__(self, simulator: WhatIfSimulator):
        """
        Initialize policy impact analyzer.
        
        Args:
            simulator: WhatIfSimulator instance
        """
        self.simulator = simulator
    
    def analyze_policy(
        self,
        current_data: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze impact of a specific policy.
        
        Args:
            current_data: Current air quality data
            policy: Policy definition with parameters
        
        Returns:
            Dictionary with policy impact analysis
        """
        try:
            # Convert policy to scenario
            scenario = self._policy_to_scenario(policy)
            
            # Run simulation
            result = self.simulator.simulate(current_data, scenario)
            
            # Add policy-specific analysis
            result["policy"] = policy
            result["policy_effectiveness"] = self._calculate_effectiveness(result)
            result["implementation_feasibility"] = self._assess_feasibility(policy)
            result["cost_benefit"] = self._estimate_cost_benefit(policy, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing policy: {e}", error=str(e))
            return {"error": str(e)}
    
    def _policy_to_scenario(self, policy: Dict[str, Any]) -> Dict[str, Any]:
        """Convert policy definition to simulation scenario."""
        scenario = {}
        
        policy_type = policy.get("type", "")
        policy_params = policy.get("parameters", {})
        
        if policy_type == "traffic_reduction":
            scenario["traffic_reduction"] = policy_params.get("reduction_percent", 0) / 100
        
        elif policy_type == "green_cover":
            scenario["green_cover_increase"] = policy_params.get("increase_percent", 0) / 100
        
        elif policy_type == "industrial_control":
            scenario["industrial_reduction"] = policy_params.get("reduction_percent", 0) / 100
        
        elif policy_type == "composite":
            # Multiple policies
            if "traffic_reduction" in policy_params:
                scenario["traffic_reduction"] = policy_params["traffic_reduction"] / 100
            if "green_cover_increase" in policy_params:
                scenario["green_cover_increase"] = policy_params["green_cover_increase"] / 100
            if "industrial_reduction" in policy_params:
                scenario["industrial_reduction"] = policy_params["industrial_reduction"] / 100
        
        return scenario
    
    def _calculate_effectiveness(self, result: Dict[str, Any]) -> str:
        """Calculate policy effectiveness rating."""
        aqi_change = result.get("aqi_change", 0)
        change_percent = result.get("change_percent", 0)
        
        if abs(change_percent) > 20:
            return "high"
        elif abs(change_percent) > 10:
            return "medium"
        elif abs(change_percent) > 5:
            return "low"
        else:
            return "minimal"
    
    def _assess_feasibility(self, policy: Dict[str, Any]) -> str:
        """Assess implementation feasibility."""
        policy_type = policy.get("type", "")
        
        feasibility_map = {
            "traffic_reduction": "medium",  # Requires infrastructure
            "green_cover": "high",  # Relatively easy
            "industrial_control": "low",  # Requires regulation
            "composite": "medium"
        }
        
        return feasibility_map.get(policy_type, "unknown")
    
    def _estimate_cost_benefit(
        self,
        policy: Dict[str, Any],
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Estimate cost-benefit ratio."""
        aqi_improvement = abs(result.get("aqi_change", 0))
        
        # Rough estimates (in relative terms)
        cost_estimates = {
            "traffic_reduction": "high",  # Infrastructure costs
            "green_cover": "medium",  # Land and maintenance
            "industrial_control": "high",  # Compliance costs
            "composite": "high"
        }
        
        policy_type = policy.get("type", "")
        estimated_cost = cost_estimates.get(policy_type, "unknown")
        
        # Benefit is AQI improvement
        benefit = "high" if aqi_improvement > 20 else "medium" if aqi_improvement > 10 else "low"
        
        return {
            "estimated_cost": estimated_cost,
            "estimated_benefit": benefit,
            "aqi_improvement": aqi_improvement,
            "cost_benefit_ratio": self._calculate_ratio(estimated_cost, benefit)
        }
    
    def _calculate_ratio(self, cost: str, benefit: str) -> str:
        """Calculate cost-benefit ratio."""
        cost_map = {"low": 1, "medium": 2, "high": 3}
        benefit_map = {"low": 1, "medium": 2, "high": 3}
        
        cost_val = cost_map.get(cost, 2)
        benefit_val = benefit_map.get(benefit, 2)
        
        ratio = benefit_val / cost_val if cost_val > 0 else 0
        
        if ratio > 1.5:
            return "favorable"
        elif ratio > 1.0:
            return "moderate"
        else:
            return "unfavorable"

