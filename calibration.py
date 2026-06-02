"""
calibration.py - Pure functions for AI-4.5 & AI-4.7 soft-coupling rule

This module contains the core decision logic for calibrating LLM estimates
against USDA FoodData Central data. These are pure functions with no external
dependencies, making them easy to test and reason about.
"""

from typing import Tuple, Optional, Literal


def pct_diff(a: float, b: float) -> float:
    """
    Calculate percentage difference between two values.
    
    Args:
        a: Reference value (LLM estimate)
        b: Comparison value (USDA value)
    
    Returns:
        Percentage difference as a decimal (0.0 to 1.0+)
    """
    if a == 0:
        return 1.0 if b else 0.0
    return abs(a - b) / abs(a)


DecideSource = Literal["llm", "usda"]
DecideReason = Literal["usda_missing", "within_25%", "deviation_too_high"]


def decide(
    llm_value: float,
    usda_value: Optional[float],
    threshold: float = 0.25
) -> Tuple[float, DecideSource, DecideReason]:
    """
    Apply soft-coupling rule to decide between LLM and USDA values.
    
    Decision logic:
    - If USDA is None → Use LLM (reason: usda_missing)
    - If |LLM - USDA| / LLM ≤ threshold → Use USDA (reason: within_25%)
    - Otherwise → Use LLM (reason: deviation_too_high)
    
    Args:
        llm_value: The LLM-estimated macro value
        usda_value: The USDA reference value (or None if not found)
        threshold: Maximum acceptable deviation (default 0.25 = 25%)
    
    Returns:
        Tuple of (final_value, source, reason)
        - final_value: The chosen value
        - source: "llm" or "usda" indicating provenance
        - reason: Why this choice was made
    """
    if usda_value is None:
        return llm_value, "llm", "usda_missing"

    if pct_diff(llm_value, usda_value) <= threshold:
        return usda_value, "usda", "within_25%"
    
    return llm_value, "llm", "deviation_too_high"


def apply_coupling_mode(
    llm_value: float,
    usda_value: Optional[float],
    coupling_mode: Literal["soft", "strict", "llm_only"] = "soft",
    threshold: float = 0.25
) -> Tuple[float, DecideSource, str]:
    """
    Apply coupling mode to decide between LLM and USDA values.
    
    Supports A/B testing different coupling strategies:
    - "soft": Use USDA if within threshold (default soft-coupling rule)
    - "strict": Always use USDA when available
    - "llm_only": Never use USDA, always trust LLM
    
    Args:
        llm_value: The LLM-estimated macro value
        usda_value: The USDA reference value (or None if not found)
        coupling_mode: One of "soft", "strict", or "llm_only"
        threshold: Threshold for soft coupling (default 0.25)
    
    Returns:
        Tuple of (final_value, source, reason)
    """
    if usda_value is None:
        return llm_value, "llm", "usda_missing"
    
    if coupling_mode == "llm_only":
        return llm_value, "llm", "llm_only_mode"
    
    if coupling_mode == "strict":
        return usda_value, "usda", "strict_mode"
    
    # Default: soft coupling
    return decide(llm_value, usda_value, threshold)
