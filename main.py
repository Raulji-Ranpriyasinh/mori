import os
import base64
import json
import requests
from typing import Dict, Any, Tuple, List, Literal
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from usda_client import search_usda_food,extract_usda_macros
from langchain_google_genai import ChatGoogleGenerativeAI
from calibration import decide, apply_coupling_mode, pct_diff
import calibration


# ENV


load_dotenv()

USDA_API_KEY = os.getenv("USDA_API")
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)


# IMAGE UTIL


def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()



# STRUCTURED LLM OUTPUT


class Component(BaseModel):
    name: str
    grams: float
    carbs_g: float
    protein_g: float
    fat_g: float
    fiber_g: float


class Meal(BaseModel):
    description: str
    portion_hint: Literal["small", "normal", "large"]
    parse_confidence: float
    caveats: List[str]
    components: List[Component]


structured_model = model.with_structured_output(Meal)


# AI-4.7 ENGINE (NO TRACE IN JSON)

# Note: pct_diff and decide are now imported from calibration.py
# These wrappers are deprecated but kept for backward compatibility


def decide(llm, usda, threshold=0.25):
    """Deprecated: use calibration.decide() instead"""
    return calibration.decide(llm, usda, threshold)



# LLM STEP


def analyze_image(image_path: str) -> Meal:
    data_uri = image_to_data_uri(image_path)

    prompt = """
Return structured meal data.
Ensure food names are CLEAN (no cooked/raw/fried annotations).
"""

    return structured_model.invoke([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": data_uri}
            ]
        }
    ])



# TRACE (SEPARATE - NO JSON IN OUTPUT)


def log_langsmith_trace(decisions: list):
    # replace with real LangSmith client in production
    print("\n[LangSmith] food.calibrate")
    print(json.dumps({
        "rule": "AI-4.7_soft_coupling",
        "decisions": decisions
    }, indent=2))



# PIPELINE


def process_image(image_path: str, photo_ref: str = None):

    meal: Meal = analyze_image(image_path)

    eaten_at = datetime.now(timezone.utc).isoformat()

    final_components = []
    trace_decisions = []

    totals = {
        "carbs_g": 0,
        "protein_g": 0,
        "fat_g": 0,
        "fiber_g": 0,
        "gl": 0.0
    }

    for c in meal.components:

        usda = search_usda_food(c.name)

        usda_macros = extract_usda_macros(usda, c.grams) if usda else None

        final = {}

        comp_out = {
            "name": c.name,
            "grams": c.grams,
            "carbs_g": {},
            "protein_g": {},
            "fat_g": {},
            "fiber_g": {}
        }

        for m in ["carbs_g", "protein_g", "fat_g", "fiber_g"]:
            val, src, reason = decide(
                getattr(c, m),
                usda_macros[m] if usda_macros else None
            )

            comp_out[m] = {
                "value": val,
                "source": src
            }

            totals[m] += val

            trace_decisions.append({
                "component": c.name,
                "macro": m,
                "llm": getattr(c, m),
                "usda": usda_macros[m] if usda_macros else None,
                "final": val,
                "source": src,
                "reason": reason
            })

        final_components.append(comp_out)

    # -------------------------
    # FINAL OUTPUT (CLEAN ONLY)
    # -------------------------

    final_payload = {
        "description": meal.description,
        "eaten_at": eaten_at,
        "photo_ref": photo_ref,
        "portion_hint": meal.portion_hint,
        "parse_confidence": meal.parse_confidence,
        "caveats": meal.caveats,

        "components": final_components,

        "totals": totals
    }

    # -------------------------
    # TRACE (SEPARATE SYSTEM)
    # -------------------------

    log_langsmith_trace(trace_decisions)

    return final_payload



# RUN


if __name__ == "__main__":

    result = process_image(
        "img/f2.png",
        photo_ref="photo_xyz123"
    )

    print("\nFINAL OUTPUT:\n")
    print(json.dumps(result, indent=2))
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n✅ Saved output to result.json")