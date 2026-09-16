# API reference

## `GET /api/v1/health`

Returns service status and the current image model status.

## `POST /api/v1/analyze`

Accepts JSON fields `nitrogen`, `phosphorus`, `potassium`, `ph`,
`organic_matter`, and `moisture`. Values are validated for physical input ranges.
The response contains `health_score`, `deficiencies`, `degradation_risks`,
`recommendations`, and `crop_suitability`.

## `POST /api/v1/analyze-image`

Accepts the same six measurements as multipart form fields and an `image` field.
Only JPEG and PNG files up to `MAX_UPLOAD_BYTES` are accepted. The image result
is currently explicitly marked `not_trained`; no nutrient value is inferred from
the image.
