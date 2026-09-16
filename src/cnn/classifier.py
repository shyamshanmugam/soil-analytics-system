from dataclasses import asdict, dataclass
from io import BytesIO

from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class ImageClassification:
    status: str
    label: str | None
    confidence: float | None
    explanation: str


def classify_soil_image(image_bytes: bytes) -> ImageClassification:
    """Explicit placeholder until a documented CNN is trained and loaded."""
    if not image_bytes:
        raise ValueError("Image content is empty")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("Uploaded file is not a valid image") from error
    return ImageClassification(
        status="not_trained",
        label=None,
        confidence=None,
        explanation="No trained CNN artefact is available; image nutrients are not inferred.",
    )


def classification_dict(image_bytes: bytes) -> dict:
    return asdict(classify_soil_image(image_bytes))
