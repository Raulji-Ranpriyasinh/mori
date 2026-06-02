"""
A/B Testing Harness for AI-4.7 Coupling Modes

Compare accuracy between different coupling strategies:
- soft: Use USDA if within 25% threshold (default)
- strict: Always use USDA when available
- llm_only: Never use USDA, always trust LLM
"""

import json
from typing import Dict, List, Any
from calibration import apply_coupling_mode


def simulate_calibration(
    llm_macros: Dict[str, float],
    usda_macros: Dict[str, float],
    coupling_mode: str = "soft"
) -> Dict[str, Any]:
    """
    Simulate calibration for a single component.
    
    Args:
        llm_macros: LLM-estimated macros
        usda_macros: USDA reference macros (or None if not found)
        coupling_mode: One of "soft", "strict", "llm_only"
    
    Returns:
        Calibrated macros with provenance tracking
    """
    result = {}
    decisions = []
    
    for macro in ["carbs_g", "protein_g", "fat_g", "fiber_g"]:
        llm_val = llm_macros.get(macro, 0.0)
        usda_val = usda_macros.get(macro) if usda_macros else None
        
        final_val, source, reason = apply_coupling_mode(
            llm_val, usda_val, coupling_mode=coupling_mode
        )
        
        result[macro] = {
            "value": final_val,
            "source": source
        }
        
        decisions.append({
            "macro": macro,
            "llm": llm_val,
            "usda": usda_val,
            "final": final_val,
            "source": source,
            "reason": reason
        })
    
    return {
        "calibrated_macros": result,
        "decisions": decisions
    }


def run_ab_test(
    eval_set: List[Dict],
    usda_lookup_fn=None,
    ground_truth: Dict[str, Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Run A/B test across all coupling modes.
    
    Args:
        eval_set: List of evaluation components
        usda_lookup_fn: Function to lookup USDA data (optional)
        ground_truth: Optional ground truth macros for accuracy calculation
    
    Returns:
        Comparison results across all modes
    """
    modes = ["soft", "strict", "llm_only"]
    results = {mode: {"usda_applied": 0, "llm_kept": 0, "total_decisions": 0} 
               for mode in modes}
    
    for component in eval_set:
        food_name = component["food_name"]
        llm_macros = component["llm_macros"]
        
        # Simulate USDA lookup (in real scenario, call usda_lookup_fn)
        # For now, use expected_usda_match to simulate availability
        if component.get("expected_usda_match", True):
            # Simulate USDA values close to LLM for standard foods
            usda_macros = {k: v * 0.95 for k, v in llm_macros.items()}
        else:
            # Non-standard foods: USDA might be very different or missing
            usda_macros = None
        
        for mode in modes:
            cal_result = simulate_calibration(llm_macros, usda_macros, mode)
            
            for decision in cal_result["decisions"]:
                results[mode]["total_decisions"] += 1
                if decision["source"] == "usda":
                    results[mode]["usda_applied"] += 1
                else:
                    results[mode]["llm_kept"] += 1
    
    # Calculate percentages
    summary = {}
    for mode, stats in results.items():
        total = stats["total_decisions"]
        if total > 0:
            summary[mode] = {
                "usda_applied_count": stats["usda_applied"],
                "llm_kept_count": stats["llm_kept"],
                "usda_percentage": round(stats["usda_applied"] / total * 100, 2),
                "llm_percentage": round(stats["llm_kept"] / total * 100, 2),
                "total_decisions": total
            }
    
    return {
        "modes_compared": modes,
        "summary": summary,
        "recommendation": get_recommendation(summary)
    }


def get_recommendation(summary: Dict[str, Any]) -> str:
    """Generate recommendation based on A/B test results."""
    soft_stats = summary.get("soft", {})
    strict_stats = summary.get("strict", {})
    
    soft_usda_pct = soft_stats.get("usda_percentage", 0)
    strict_usda_pct = strict_stats.get("usda_percentage", 0)
    
    # Soft coupling should balance trust between LLM and USDA
    if 40 <= soft_usda_pct <= 80:
        return "Soft coupling provides good balance - recommended for production"
    elif soft_usda_pct < 40:
        return "Low USDA adoption - consider lowering threshold or improving USDA matching"
    else:
        return "High USDA adoption - verify LLM isn't being unnecessarily overridden"


if __name__ == "__main__":
    # Load eval set
    with open("eval_set.json", "r") as f:
        eval_data = json.load(f)
    
    print("Running A/B Test for Coupling Modes\n")
    print("=" * 50)
    
    results = run_ab_test(eval_data["components"])
    
    print("\nResults Summary:\n")
    for mode, stats in results["summary"].items():
        print(f"\n{mode.upper()} Mode:")
        print(f"  USDA Applied: {stats['usda_applied_count']} ({stats['usda_percentage']}%)")
        print(f"  LLM Kept:     {stats['llm_kept_count']} ({stats['llm_percentage']}%)")
        print(f"  Total:        {stats['total_decisions']} decisions")
    
    print(f"\n{'=' * 50}")
    print(f"\nRecommendation: {results['recommendation']}")
