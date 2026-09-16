from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_is_explicit_about_model_state():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["model_status"] == "image_classifier_not_trained"


def test_structured_analysis_reports_low_nutrients():
    response = client.post(
        "/api/v1/analyze",
        json={"nitrogen": 10, "phosphorus": 5, "potassium": 50, "ph": 6.8, "organic_matter": 2, "moisture": 25},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["health_score"] < 100
    assert "nitrogen" in body["deficiencies"]
    assert body["image_classification"] is None


def test_image_endpoint_rejects_unsupported_type():
    payload = {"nitrogen": 40, "phosphorus": 20, "potassium": 140, "ph": 6.5, "organic_matter": 2, "moisture": 25}
    response = client.post(
        "/api/v1/analyze-image",
        data=payload,
        files={"image": ("soil.txt", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 415


def test_image_endpoint_rejects_invalid_image_bytes():
    payload = {"nitrogen": 40, "phosphorus": 20, "potassium": 140, "ph": 6.5, "organic_matter": 2, "moisture": 25}
    response = client.post(
        "/api/v1/analyze-image",
        data=payload,
        files={"image": ("soil.png", b"not-an-image", "image/png")},
    )
    assert response.status_code == 400
