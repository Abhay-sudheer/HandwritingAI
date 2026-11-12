from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


@dataclass
class PreparedImage:
    image: Image.Image
    lines: list[Image.Image]
    diagnostics: list[str]
    paragraph_breaks: set[int]


def prepare_image(data: bytes, mode: str = "auto", max_dimension: int = 4000) -> PreparedImage:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc
    if image.width * image.height > 30_000_000:
        raise ValueError("Image dimensions are too large (maximum 30 megapixels).")
    if max(image.size) > max_dimension:
        scale = max_dimension / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold tolerates uneven lighting; Otsu provides a second path for clean scans.
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
    otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    mask = adaptive if np.count_nonzero(adaptive) < gray.size * .45 else otsu
    kernel_width = max(15, image.width // 45)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1)))
    projection = np.count_nonzero(joined, axis=1)
    active = projection > max(2, image.width * .006)
    # Bridge small vertical gaps to keep tall glyphs and slightly slanted lines together.
    gap = max(2, image.height // 180)
    for i in range(1, gap + 1):
        active[i:] |= active[:-i]
    groups: list[tuple[int, int]] = []
    start = None
    for y, value in enumerate(active):
        if value and start is None: start = y
        if not value and start is not None:
            if y - start >= max(5, image.height // 150): groups.append((start, y))
            start = None
    if start is not None and len(active) - start >= max(5, image.height // 150): groups.append((start, len(active)))
    lines = []
    line_spans = []
    pad = max(4, round(image.height * .008))
    for top, bottom in groups:
        top, bottom = max(0, top-pad), min(image.height, bottom+pad)
        if bottom-top < 8: continue
        cropmask = mask[top:bottom, :]
        xs = np.where(np.any(cropmask > 0, axis=0))[0]
        if len(xs) < 2: continue
        left, right = max(0, int(xs[0])-pad), min(image.width, int(xs[-1])+1+pad)
        lines.append(image.crop((left, top, right, bottom)))
        line_spans.append((top, bottom))
    diagnostics = ["Orientation corrected using embedded image metadata when available.", "Noise reduced and contrast-adaptive line segmentation applied."]
    paragraph_breaks: set[int] = set()
    if mode == "single" or not lines:
        lines = [image]
        line_spans = [(0, image.height)]
        diagnostics.append("Single-region fallback used; page layout and paragraph boundaries may not be preserved.")
    else:
        gaps = [max(0, line_spans[i][0] - line_spans[i - 1][1]) for i in range(1, len(line_spans))]
        typical_gap = float(np.median(gaps)) if gaps else 0.0
        if typical_gap > 0:
            paragraph_breaks = {i for i, gap_value in enumerate(gaps, start=1) if gap_value > typical_gap * 1.8}
        if paragraph_breaks:
            diagnostics.append(f"Estimated {len(paragraph_breaks)} paragraph break(s) from wider vertical spacing.")
        diagnostics.append(f"Detected {len(lines)} text line region(s) in top-to-bottom order.")
    return PreparedImage(image=image, lines=lines, diagnostics=diagnostics, paragraph_breaks=paragraph_breaks)
