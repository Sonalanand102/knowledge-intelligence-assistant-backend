from __future__ import annotations

from math import sqrt
from typing import Any


def get_bbox(element) -> dict[str, float] | None:
    bbox = element.metadata.get("bbox")

    if not bbox:
        return None

    if isinstance(bbox, dict):
        required = {"x0", "y0", "x1", "y1"}

        if required.issubset(bbox):
            return {
                "x0": float(bbox["x0"]),
                "y0": float(bbox["y0"]),
                "x1": float(bbox["x1"]),
                "y1": float(bbox["y1"]),
            }

        if {"left", "top", "width", "height"}.issubset(bbox):
            return {
                "x0": float(bbox["left"]),
                "y0": float(bbox["top"]),
                "x1": float(
                    bbox["left"] + bbox["width"]
                ),
                "y1": float(
                    bbox["top"] + bbox["height"]
                ),
            }

    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        return {
            "x0": float(bbox[0]),
            "y0": float(bbox[1]),
            "x1": float(bbox[2]),
            "y1": float(bbox[3]),
        }

    return None


def center(
    bbox: dict[str, float],
) -> tuple[float, float]:
    return (
        (bbox["x0"] + bbox["x1"]) / 2,
        (bbox["y0"] + bbox["y1"]) / 2,
    )


def euclidean_distance(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
) -> float:
    center_a = center(bbox_a)
    center_b = center(bbox_b)

    return sqrt(
        (center_a[0] - center_b[0]) ** 2
        + (center_a[1] - center_b[1]) ** 2
    )


def vertical_gap(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
) -> float:
    if bbox_a["y1"] < bbox_b["y0"]:
        return bbox_b["y0"] - bbox_a["y1"]

    if bbox_b["y1"] < bbox_a["y0"]:
        return bbox_a["y0"] - bbox_b["y1"]

    return 0.0


def horizontal_gap(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
) -> float:
    if bbox_a["x1"] < bbox_b["x0"]:
        return bbox_b["x0"] - bbox_a["x1"]

    if bbox_b["x1"] < bbox_a["x0"]:
        return bbox_a["x0"] - bbox_b["x1"]

    return 0.0


def is_spatially_near(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
    max_vertical_gap: float = 80.0,
    max_horizontal_gap: float = 80.0,
) -> bool:
    return (
        vertical_gap(bbox_a, bbox_b)
        <= max_vertical_gap
        and horizontal_gap(bbox_a, bbox_b)
        <= max_horizontal_gap
    )


def spatial_metadata(
    bbox_a: dict[str, float],
    bbox_b: dict[str, float],
) -> dict[str, Any]:
    return {
        "euclidean_distance": euclidean_distance(
            bbox_a,
            bbox_b,
        ),
        "vertical_gap": vertical_gap(
            bbox_a,
            bbox_b,
        ),
        "horizontal_gap": horizontal_gap(
            bbox_a,
            bbox_b,
        ),
    }