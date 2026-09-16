from pydantic import BaseModel, Field


class SoilTest(BaseModel):
    nitrogen: float = Field(ge=0, le=1000)
    phosphorus: float = Field(ge=0, le=1000)
    potassium: float = Field(ge=0, le=2000)
    ph: float = Field(ge=0, le=14)
    organic_matter: float = Field(ge=0, le=100)
    moisture: float = Field(ge=0, le=100)


class AnalysisResponse(BaseModel):
    health_score: float
    deficiencies: list[str]
    degradation_risks: list[str]
    recommendations: list[dict[str, str]]
    crop_suitability: list[dict[str, str | float | bool]]
    image_classification: dict | None = None
