"""Turn a native PDF into a "scanned" one: raster image, skew, blur, noise, JPEG artefacts."""

import io
import random
from enum import StrEnum

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageFilter
from pydantic import BaseModel, ConfigDict
from reportlab import rl_config
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

rl_config.invariant = 1


class ScanLevel(StrEnum):
    CLEAN = "clean"
    LIGHT = "light"
    HEAVY = "heavy"


class _Degradation(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_skew_deg: float
    blur_radius: float
    noise_sigma: float
    contrast: float
    jpeg_quality: int


LEVELS = {
    ScanLevel.CLEAN: _Degradation(
        max_skew_deg=0, blur_radius=0, noise_sigma=0, contrast=1.0, jpeg_quality=85
    ),
    ScanLevel.LIGHT: _Degradation(
        max_skew_deg=0.8, blur_radius=0.4, noise_sigma=5, contrast=0.9, jpeg_quality=60
    ),
    ScanLevel.HEAVY: _Degradation(
        max_skew_deg=2.5, blur_radius=0.9, noise_sigma=14, contrast=0.75, jpeg_quality=35
    ),
}

DPI = 150


def degrade_pdf(pdf: bytes, level: ScanLevel, rng: random.Random) -> bytes:
    """Rasterize every page and re-embed it as a degraded grayscale JPEG (no text layer)."""
    spec = LEVELS[level]
    out = io.BytesIO()
    canvas = Canvas(out, pagesize=A4)
    for page in pdfium.PdfDocument(pdf):
        image = page.render(scale=DPI / 72, grayscale=True).to_pil().convert("L")
        image = _degrade(image, spec, rng)
        jpeg = io.BytesIO()
        image.save(jpeg, format="JPEG", quality=spec.jpeg_quality)
        canvas.drawImage(ImageReader(io.BytesIO(jpeg.getvalue())), 0, 0, *A4)
        canvas.showPage()
    canvas.save()
    return out.getvalue()


def _degrade(image: Image.Image, spec: _Degradation, rng: random.Random) -> Image.Image:
    if spec.max_skew_deg:
        angle = rng.uniform(-spec.max_skew_deg, spec.max_skew_deg)
        image = image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=255)
    if spec.blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(spec.blur_radius))
    pixels = np.asarray(image, dtype=np.float32)
    # Lower contrast around mid-grey (faded toner), then add sensor noise.
    pixels = 128 + (pixels - 128) * spec.contrast
    if spec.noise_sigma:
        noise_rng = np.random.default_rng(rng.getrandbits(64))
        pixels += noise_rng.normal(0, spec.noise_sigma, pixels.shape)
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="L")
