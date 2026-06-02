# usda_client.py
import os
import requests
from dotenv import load_dotenv
load_dotenv()
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"
USDA_API_KEY = os.getenv("USDA_API")


def search_usda_food(query: str):
    res = requests.get(
        f"{USDA_BASE_URL}/foods/search",
        params={
            "api_key": USDA_API_KEY,
            "query": query,
            "pageSize": 1
        }
    )
    res.raise_for_status()
    foods = res.json().get("foods", [])
    return foods[0] if foods else None


def extract_usda_macros(food, grams: float):
    if not food:
        return None

    nutrients = food.get("foodNutrients", [])
    factor = grams / 100

    def find(name):
        for n in nutrients:
            if name.lower() in n.get("nutrientName", "").lower():
                return n.get("value", 0.0)
        return 0.0

    return {
        "carbs_g": find("carbohydrate") * factor,
        "protein_g": find("protein") * factor,
        "fat_g": find("total lipid") * factor,
        "fiber_g": find("fiber") * factor,
    }