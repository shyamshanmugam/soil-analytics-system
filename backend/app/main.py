import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .schemas import AnalysisResponse, SoilTest
from .services import analyze
from src.cnn.classifier import classify_soil_image

app = FastAPI(
    title="Soil Analytics API",
    version="0.1.0",
    description="Separate image-classification and laboratory-data soil analysis paths.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "5242880"))
ALLOWED_TYPES = {"image/jpeg", "image/png"}


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_status": "image_classifier_not_trained"}


@app.post("/api/v1/analyze", response_model=AnalysisResponse)
def analyze_structured(test: SoilTest) -> dict:
    return analyze(test)


@app.post("/api/v1/analyze-image", response_model=AnalysisResponse)
async def analyze_image(
    nitrogen: float = Form(...),
    phosphorus: float = Form(...),
    potassium: float = Form(...),
    ph: float = Form(...),
    organic_matter: float = Form(...),
    moisture: float = Form(...),
    image: UploadFile = File(...),
) -> dict:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported")
    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Image content is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the configured size limit")
    try:
        classify_soil_image(content)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    try:
        test = SoilTest(
            nitrogen=nitrogen,
            phosphorus=phosphorus,
            potassium=potassium,
            ph=ph,
            organic_matter=organic_matter,
            moisture=moisture,
        )
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors()) from error
    return analyze(test, content)
