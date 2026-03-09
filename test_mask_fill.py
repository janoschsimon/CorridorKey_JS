"""
Test script: fill holes in alpha hint masks and show side-by-side comparison.
Usage: uv run python test_mask_fill.py <AlphaHint_folder>
"""
import os
import sys
import cv2
import numpy as np


def fill_mask_holes(mask_u8: np.ndarray) -> np.ndarray:
    """Close gaps with large kernel morphology, then fill internal holes."""
    # 1. Big close to bridge gaps between body parts (arms, legs etc)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    # 2. Fill remaining internal holes via contour fill
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask_u8)
    cv2.drawContours(result, contours, -1, 255, thickness=cv2.FILLED)
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python test_mask_fill.py <AlphaHint_folder>")
        sys.exit(1)

    folder = sys.argv[1]
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".png")])[:10]

    out_dir = os.path.join(folder, "_test_fill_sbs")
    os.makedirs(out_dir, exist_ok=True)

    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalize to uint8 0/255
        if img.max() <= 1:
            img = (img * 255).astype(np.uint8)
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        filled = fill_mask_holes(binary)

        # Side by side
        gap = np.zeros((binary.shape[0], 10), np.uint8)
        sbs = np.hstack([binary, gap, filled])

        # Add labels
        sbs_bgr = cv2.cvtColor(sbs, cv2.COLOR_GRAY2BGR)
        cv2.putText(sbs_bgr, "ORIGINAL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
        cv2.putText(sbs_bgr, "FILLED", (binary.shape[1] + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 2)

        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, sbs_bgr)
        print(f"Saved: {out_path}")

    print(f"\nDone! Check: {out_dir}")


if __name__ == "__main__":
    main()
