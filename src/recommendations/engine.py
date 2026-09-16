def recommendations(deficiencies: list[str]) -> list[dict[str, str]]:
    advice = {
        "nitrogen": ("Nitrogen support", "Use a soil-test-guided nitrogen source and split applications."),
        "phosphorus": ("Phosphorus support", "Use a measured phosphorus amendment near planting."),
        "potassium": ("Potassium support", "Apply a potassium source according to a local agronomist's dose."),
        "organic_matter": ("Organic matter", "Add mature compost or well-decomposed organic material."),
        "moisture": ("Moisture management", "Use mulch and adjust irrigation after checking drainage."),
        "acidic_pH": ("Acidic soil", "Confirm with a laboratory test before applying agricultural lime."),
        "alkaline_pH": ("Alkaline soil", "Use organic matter and a locally recommended amendment plan."),
    }
    return [{"title": advice[item][0], "guidance": advice[item][1]} for item in deficiencies if item in advice]


def crop_suitability(ph: float, moisture: float) -> list[dict[str, str | float]]:
    candidates = [
        ("rice", 5.5 <= ph <= 7.5 and moisture >= 30),
        ("wheat", 6.0 <= ph <= 7.5 and 20 <= moisture <= 45),
        ("millet", 5.5 <= ph <= 8.0 and moisture < 35),
    ]
    return [{"crop": crop, "suitable": suitable} for crop, suitable in candidates]
