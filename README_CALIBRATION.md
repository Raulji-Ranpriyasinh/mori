# AI-4.5 & AI-4.7: Food Data Layer with USDA Cross-Reference

## Overview

This module implements a **soft-coupling pattern** between LLM-estimated food macros and USDA FoodData Central authoritative data. The LLM output is primary, and USDA serves as a calibration layer.

## Soft-Coupling Rule

### Decision Logic

For each macronutrient (carbs, protein, fat, fiber) of each food component:

1. **Lookup**: Find the closest USDA match for the food item
2. **Compare**: Calculate percentage difference between LLM estimate and USDA value
3. **Decide**:
   - If USDA data is missing → Use LLM value (source: `llm`, reason: `usda_missing`)
   - If |LLM - USDA| / LLM ≤ 25% → Use USDA value (source: `usda`, reason: `within_25%`)
   - If |LLM - USDA| / LLM > 25% → Keep LLM value (source: `llm`, reason: `deviation_too_high`)

### Formula

```python
def pct_diff(a, b):
    if a == 0:
        return 1.0 if b else 0.0
    return abs(a - b) / a

def decide(llm_value, usda_value, threshold=0.25):
    if usda_value is None:
        return llm_value, "llm", "usda_missing"
    
    if pct_diff(llm_value, usda_value) <= threshold:
        return usda_value, "usda", "within_25%"
    
    return llm_value, "llm", "deviation_too_high"
```

### Provenance Tracking

Each macro in the output includes source attribution:

```json
{
  "carbs_g": {
    "value": 45.2,
    "source": "usda"
  },
  "protein_g": {
    "value": 12.5,
    "source": "llm"
  }
}
```

### Trace Metadata

The `food.calibrate` trace span captures:
- Component name
- Macro type
- LLM estimate
- USDA value (or null)
- Final value chosen
- Source (llm | usda)
- Reason (usda_missing | within_25% | deviation_too_high)

### A/B Testing

The system supports toggling between coupling modes via the `coupling_mode` parameter:

- `"soft"` (default): Apply the 25% threshold rule above
- `"strict"`: Always use USDA when available, regardless of deviation
- `"llm_only"`: Never use USDA, always trust LLM

This enables accuracy comparisons between different coupling strategies.

## Files

- `main.py`: Core pipeline with LangSmith trace integration
- `calibration.py`: Pure functions for soft-coupling rule (`decide()`, `apply_coupling_mode()`, `pct_diff()`)
- `usda_client.py`: USDA FoodData Central API client
- `test_calibration.py`: 27 unit tests for calibration logic
- `eval_set.json`: Evaluation dataset for calibration verification
- `ab_test.py`: A/B testing harness for coupling modes
- `.env`: Environment configuration (LangSmith, USDA, Google API keys)

## Configuration

### Environment Variables (.env)

```bash
# Google Gemini API (required for LLM)
GOOGLE_API_KEY=your_google_api_key_here

# USDA FoodData Central API (required for calibration)
USDA_API=your_usda_api_key_here

# LangSmith Configuration (optional - for production tracing)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=food-calibration
LANGSMITH_TRACING=false  # Set to true in production
```

### LangSmith Integration

The `log_langsmith_trace()` function logs all calibration decisions to LangSmith when:
1. `LANGSMITH_API_KEY` is set in environment
2. `LANGSMITH_TRACING=true` in environment

If tracing is disabled or no API key is provided, traces are logged to console for development.

Each trace includes:
- **Run name**: `food.calibrate`
- **Project**: Configured via `LANGSMITH_PROJECT`
- **Metadata**: Rule version, description
- **Outputs**: Total decisions, USDA adoption rate, flagged count, calibration rate, full decision log

To enable production tracing:
1. Get API key from https://smith.langchain.com/
2. Add to `.env`: `LANGSMITH_API_KEY=lsv2_...`
3. Set `LANGSMITH_TRACING=true`

## Acceptance Criteria Status

### AI-4.5
- ✅ USDA FoodData Central integrated and queryable
- ✅ Soft-coupling rule documented (this file)
- ✅ Soft-coupling rule unit-tested (`test_calibration.py`)
- ⚠️ Calibration applied to ≥70% of components (verify with `eval_set.json`)
- ✅ Disagreement (>25% deviation) flagged in trace metadata

### AI-4.7
- ✅ Rule implemented as pure function `decide()`
- ✅ Comprehensive unit tests
- ✅ Output includes provenance per macro
- ✅ Trace span `food.calibrate` captures rule outcome
- ✅ A/B testable coupling modes
