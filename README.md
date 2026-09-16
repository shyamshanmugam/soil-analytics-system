# AI-Powered Soil Analytics System

An internship-project starter for soil image classification and structured laboratory
soil analysis. The repository intentionally keeps the two modelling paths separate:
the image dataset supports soil-type classification, while N/P/K, pH, moisture, and
organic-matter analysis requires structured laboratory data.

## Repository audit

The initial repository contained only this README and no Week 3 implementation,
notebooks, datasets, or application code. No experimental metrics or trained models
were available to reuse. This first implementation therefore provides a runnable
baseline and clearly labels demo and untrained behaviour.

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn backend.app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the API documentation. The health endpoint is
available at `GET /api/v1/health`, and soil analysis is available at
`POST /api/v1/analyze` with a JSON body. Image upload uses
`POST /api/v1/analyze-image` as multipart form data.

To run the API in Docker:

```powershell
docker compose up --build
```

## Structured analysis example

```json
{
  "nitrogen": 42,
  "phosphorus": 18,
  "potassium": 160,
  "ph": 6.8,
  "organic_matter": 2.4,
  "moisture": 28
}
```

These values are user-supplied measurements. Files under `data/sample/` are
synthetic demonstrations and must not be presented as real agricultural results.

## Project structure

- `backend/app/`: FastAPI routes, validation schemas, and service orchestration.
- `src/`: reusable soil analysis, recommendations, and image-classification baseline.
- `frontend/`: small Next.js farmer-facing interface.
- `notebooks/`: supplied Week 3 dataset-cleaning and EfficientNetB0 Colab workflow.
- `tests/`: API and domain tests.
- `docs/`: audit, architecture, data flow, limitations, and run instructions.

The frontend accepts laboratory values and an optional JPEG/PNG soil photograph.
On mobile browsers, the image field requests the rear camera when supported.

## Important limitations

The repository does not contain a trained CNN, real structured soil dataset, or
validated agricultural benchmark. Image classification currently returns an explicit
`not_trained` result; it never infers nutrient levels from an image. Replace the
baseline services with trained artefacts only after documenting their training data
and measured evaluation results.
