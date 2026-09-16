from dataclasses import dataclass


@dataclass(frozen=True)
class SoilAnalysis:
    health_score: float
    deficiencies: list[str]
    risks: list[str]


def analyze_soil(
    nitrogen: float,
    phosphorus: float,
    potassium: float,
    ph: float,
    organic_matter: float,
    moisture: float,
) -> SoilAnalysis:
    """Return a transparent rule-based baseline, not a trained ML prediction."""
    score = 100.0
    deficiencies: list[str] = []
    risks: list[str] = []

    for value, low, label, penalty in (
        (nitrogen, 40, "nitrogen", 15),
        (phosphorus, 20, "phosphorus", 12),
        (potassium, 140, "potassium", 12),
        (organic_matter, 2, "organic_matter", 12),
        (moisture, 20, "moisture", 8),
    ):
        if value < low:
            deficiencies.append(label)
            score -= penalty

    if ph < 5.5:
        deficiencies.append("acidic_pH")
        score -= 12
    elif ph > 8.0:
        deficiencies.append("alkaline_pH")
        score -= 12

    if moisture < 15:
        risks.append("drought_stress")
    if moisture > 45:
        risks.append("waterlogging")
    if organic_matter < 1.0:
        risks.append("organic_matter_degradation")

    return SoilAnalysis(
        health_score=round(max(0.0, min(100.0, score)), 1),
        deficiencies=deficiencies,
        risks=risks,
    )
