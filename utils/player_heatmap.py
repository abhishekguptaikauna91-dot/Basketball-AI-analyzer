"""Create an image-space player activity heatmap from tracking CSV rows."""
from pathlib import Path

import cv2
import numpy as np


def create_player_heatmap(rows, background_frame, output_path: Path):
    """Save a heatmap of observed player foot positions over a video frame.

    This is intentionally image-space: without court calibration it must not be
    interpreted as a top-down court map or distance-based movement statistic.
    """
    if background_frame is None or background_frame.size == 0:
        raise ValueError("A valid video frame is required as the heatmap background")
    height, width = background_frame.shape[:2]
    density = np.zeros((height, width), dtype=np.float32)
    point_count = 0
    for row in rows:
        try:
            x = int(round(float(row["center_x_px"])))
            y = int(round(float(row["foot_y_px"])))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(density, (x, y), 2, 1.0, thickness=-1, lineType=cv2.LINE_AA)
            point_count += 1

    density = cv2.GaussianBlur(density, (0, 0), sigmaX=max(8, width / 100), sigmaY=max(8, height / 100))
    peak = float(np.percentile(density[density > 0], 99)) if np.any(density > 0) else 0.0
    if peak > 0:
        intensity = np.clip(density / peak, 0, 1)
        intensity = np.power(intensity, .65)
        heat = cv2.applyColorMap(np.uint8(intensity * 255), cv2.COLORMAP_TURBO)
        # Keep most of the source frame visible; alpha grows at the hotspots.
        alpha = (intensity * .72)[..., None]
        background = cv2.convertScaleAbs(background_frame, alpha=.55, beta=0)
        result = np.uint8(background * (1 - alpha) + heat * alpha)
    else:
        result = cv2.convertScaleAbs(background_frame, alpha=.55, beta=0)

    cv2.rectangle(result, (18, 16), (min(width - 1, 770), 86), (12, 18, 27), -1)
    cv2.putText(result, "PLAYER ACTIVITY HEATMAP", (34, 47), cv2.FONT_HERSHEY_DUPLEX,
                .8, (245, 248, 252), 1, cv2.LINE_AA)
    subtitle = f"Image coordinates | {point_count:,} tracked positions | not court-calibrated"
    cv2.putText(result, subtitle, (34, 72), cv2.FONT_HERSHEY_SIMPLEX,
                .48, (190, 204, 218), 1, cv2.LINE_AA)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError(f"Failed to write player heatmap: {output_path}")
    return {"path": str(output_path), "positions": point_count,
            "coordinate_system": "video_image_pixels", "court_calibrated": False}
