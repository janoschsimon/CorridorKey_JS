"""
generate_alpha_hints.py
-----------------------
Full pipeline: MP4/frame-folder → MatAnyone2 alpha matte → coarse AlphaHint PNG sequence.

Runs under MatAnyone2's venv:
    MatAnyone2/.venv/Scripts/python.exe generate_alpha_hints.py -i <video> -o <alphahint_dir>

Steps:
  1. Extract frame 1, auto-generate chroma key mask (HSV green range)
  2. Run MatAnyone2 inference → per-frame alpha PNGs
  3. Post-process: dilate + gaussian blur → coarse masks CorridorKey expects
  4. Save to output AlphaHint folder
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image

# ── MatAnyone2 imports ────────────────────────────────────────────────────────
MATANYONE2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MatAnyone2")
sys.path.insert(0, MATANYONE2_DIR)

from hugging_face.tools.download_util import load_file_from_url
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import get_default_device, safe_autocast_decorator
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.inference_utils import gen_dilate, gen_erosion, read_frame_from_videos


# ── Chroma key ────────────────────────────────────────────────────────────────

def generate_chroma_mask(frame_bgr: np.ndarray, hsv_lower=(35, 40, 40), hsv_upper=(90, 255, 255)) -> np.ndarray:
    """Return 8-bit mask where 255=foreground, 0=green background."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.bitwise_not(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ── Post-process: make mask coarse for CorridorKey ───────────────────────────

def coarsen_mask(mask_u8: np.ndarray, dilate_px: int, blur_px: int) -> np.ndarray:
    """Dilate + gaussian blur → coarse/blurry mask CorridorKey expects."""
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask_u8 = cv2.dilate(mask_u8, kernel)
    if blur_px > 0:
        blur_px = blur_px if blur_px % 2 == 1 else blur_px + 1  # must be odd
        mask_u8 = cv2.GaussianBlur(mask_u8, (blur_px, blur_px), 0)
    return mask_u8


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.inference_mode()
@safe_autocast_decorator()
def run(args):
    device = get_default_device()
    print(f"[AlphaHint] Device: {device}")

    # 1. Load model
    ckpt_url = "https://github.com/pq-yang/MatAnyone2/releases/download/v1.0.0/matanyone2.pth"
    ckpt_path = load_file_from_url(ckpt_url, os.path.join(MATANYONE2_DIR, "pretrained_models"))
    model = get_matanyone2_model(ckpt_path, device)
    processor = InferenceCore(model, cfg=model.cfg)

    # 2. Load frames
    vframes, fps, length, video_name = read_frame_from_videos(args.input)
    n_warmup = 10
    repeated = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    vframes = torch.cat([repeated, vframes], dim=0).float()
    length += n_warmup

    # 3. Chroma key mask for frame 1
    first_bgr_path = args.input if os.path.isdir(args.input) else None
    if first_bgr_path:
        first_frame_file = sorted(os.listdir(first_bgr_path))[0]
        first_bgr = cv2.imread(os.path.join(first_bgr_path, first_frame_file))
    else:
        cap = cv2.VideoCapture(args.input)
        ret, first_bgr = cap.read()
        cap.release()
        assert ret, "Could not read first frame from video."

    print("[AlphaHint] Generating chroma key mask for frame 1...")
    chroma_mask = generate_chroma_mask(
        first_bgr,
        hsv_lower=tuple(args.hsv_lower),
        hsv_upper=tuple(args.hsv_upper),
    )
    fg_pct = np.count_nonzero(chroma_mask) / chroma_mask.size * 100
    print(f"[AlphaHint] Foreground coverage: {fg_pct:.1f}%")

    mask_t = torch.from_numpy(chroma_mask).float().to(device)
    objects = [1]

    # 4. MatAnyone2 inference
    os.makedirs(args.output, exist_ok=True)
    print(f"[AlphaHint] Running MatAnyone2 on {length - n_warmup} frames...")

    frame_idx = 0
    for ti in tqdm.tqdm(range(length), desc="MatAnyone2"):
        image = vframes[ti]
        image_in = (image / 255.).float().to(device)

        if ti == 0:
            output_prob = processor.step(image_in, mask_t, objects=objects)
            output_prob = processor.step(image_in, first_frame_pred=True)
        elif ti <= n_warmup:
            output_prob = processor.step(image_in, first_frame_pred=True)
        else:
            output_prob = processor.step(image_in)

        mask_t = processor.output_prob_to_mask(output_prob)

        if ti <= (n_warmup - 1):
            continue  # skip warmup frames

        pha = mask_t.cpu().numpy()
        pha_u8 = np.clip(pha * 255, 0, 255).astype(np.uint8)

        # 5. Coarsen for CorridorKey
        pha_coarse = coarsen_mask(pha_u8, dilate_px=args.dilate, blur_px=args.blur)

        out_path = os.path.join(args.output, f"{frame_idx:04d}.png")
        cv2.imwrite(out_path, pha_coarse)
        frame_idx += 1

    print(f"[AlphaHint] Done! {frame_idx} frames saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AlphaHint masks via MatAnyone2")
    parser.add_argument("-i", "--input", required=True, help="Input video (mp4/mov/avi) or frame folder")
    parser.add_argument("-o", "--output", required=True, help="Output AlphaHint folder")
    parser.add_argument("--dilate", type=int, default=15, help="Dilate radius on output alpha (px). Default: 15")
    parser.add_argument("--blur", type=int, default=31, help="Gaussian blur radius on output alpha (px). Default: 31")
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=[35, 40, 40], metavar=("H", "S", "V"), help="HSV lower bound for green. Default: 35 40 40")
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=[90, 255, 255], metavar=("H", "S", "V"), help="HSV upper bound for green. Default: 90 255 255")
    args = parser.parse_args()
    run(args)
