"""
Direct inference runner — bypasses interactive input() prompts.
Called by the GUI with all settings as CLI arguments.
Usage: uv run python run_inference_direct.py --gamma s --despill 5 --despeckle 1 --despeckle_size 400 --refiner 1.0
"""
import argparse
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from clip_manager import scan_clips, is_image_file, OUTPUT_DIR
from device_utils import resolve_device

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", choices=["s", "l"], default="s")
    parser.add_argument("--despill", type=int, default=5)
    parser.add_argument("--despeckle", type=int, default=1)
    parser.add_argument("--despeckle_size", type=int, default=400)
    parser.add_argument("--refiner", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shot", default=None, help="Only process this specific shot folder name")
    args = parser.parse_args()

    user_input_is_linear = args.gamma == "l"
    despill_strength = max(0, min(10, args.despill)) / 10.0
    auto_despeckle = bool(args.despeckle)
    despeckle_size = args.despeckle_size
    refiner_scale = args.refiner

    clips = scan_clips()
    ready_clips = [c for c in clips if c.input_asset and c.alpha_asset]
    if args.shot:
        ready_clips = [c for c in ready_clips if c.name == args.shot]

    if not ready_clips:
        print("No clips ready for inference.")
        sys.exit(1)

    print(f"Found {len(ready_clips)} clip(s) ready.", flush=True)
    print(f"Settings: gamma={'linear' if user_input_is_linear else 'sRGB'}, despill={args.despill}/10, despeckle={auto_despeckle}@{despeckle_size}px, refiner={refiner_scale}", flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}", flush=True)
    print("Loading model...", flush=True)

    from CorridorKeyModule.backend import create_engine
    engine = create_engine(device=device)

    exr_flags = [
        cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF,
        cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
    ]

    for clip in ready_clips:
        print(f"Processing: {clip.name}", flush=True)

        clip_out_root = os.path.join(clip.root_path, "Output")
        fg_dir = os.path.join(clip_out_root, "FG")
        matte_dir = os.path.join(clip_out_root, "Matte")
        comp_dir = os.path.join(clip_out_root, "Comp")
        proc_dir = os.path.join(clip_out_root, "Processed")

        for d in [fg_dir, matte_dir, comp_dir, proc_dir]:
            os.makedirs(d, exist_ok=True)

        num_frames = min(clip.input_asset.frame_count, clip.alpha_asset.frame_count)
        input_files = sorted([f for f in os.listdir(clip.input_asset.path) if is_image_file(f)])
        alpha_files = sorted([f for f in os.listdir(clip.alpha_asset.path) if is_image_file(f)])

        for i in range(num_frames):
            print(f"FRAME {i+1}/{num_frames}", flush=True)

            # Read input
            fpath = os.path.join(clip.input_asset.path, input_files[i])
            input_stem = os.path.splitext(input_files[i])[0]
            is_exr = fpath.lower().endswith(".exr")

            if is_exr:
                img_raw = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
                if img_raw is None:
                    continue
                img_srgb = np.maximum(cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB), 0.0)
                input_is_linear = True
            else:
                img_bgr = cv2.imread(fpath)
                if img_bgr is None:
                    continue
                img_srgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                input_is_linear = user_input_is_linear

            # Read mask
            apath = os.path.join(clip.alpha_asset.path, alpha_files[i])
            mask_in = cv2.imread(apath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
            if mask_in is None:
                continue

            if mask_in.ndim == 3:
                mask_linear = mask_in[:, :, 0]
            else:
                mask_linear = mask_in

            if mask_linear.dtype == np.uint8:
                mask_linear = mask_linear.astype(np.float32) / 255.0
            elif mask_linear.dtype == np.uint16:
                mask_linear = mask_linear.astype(np.float32) / 65535.0
            else:
                mask_linear = mask_linear.astype(np.float32)

            if mask_linear.shape[:2] != img_srgb.shape[:2]:
                mask_linear = cv2.resize(mask_linear, (img_srgb.shape[1], img_srgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Inference
            res = engine.process_frame(
                img_srgb, mask_linear,
                input_is_linear=input_is_linear,
                fg_is_straight=True,
                despill_strength=despill_strength,
                auto_despeckle=auto_despeckle,
                despeckle_size=despeckle_size,
                refiner_scale=refiner_scale,
            )

            pred_fg = res["fg"]
            pred_alpha = res["alpha"]

            # Save FG
            cv2.imwrite(os.path.join(fg_dir, f"{input_stem}.exr"), cv2.cvtColor(pred_fg, cv2.COLOR_RGB2BGR), exr_flags)

            # Save Matte
            if pred_alpha.ndim == 3:
                pred_alpha = pred_alpha[:, :, 0]
            cv2.imwrite(os.path.join(matte_dir, f"{input_stem}.exr"), pred_alpha, exr_flags)

            # Save Comp preview
            comp_srgb = res["comp"]
            comp_bgr = cv2.cvtColor((np.clip(comp_srgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(comp_dir, f"{input_stem}.png"), comp_bgr)

            # Save Processed RGBA
            if "processed" in res:
                proc_bgra = cv2.cvtColor(res["processed"], cv2.COLOR_RGBA2BGRA)
                cv2.imwrite(os.path.join(proc_dir, f"{input_stem}.exr"), proc_bgra, exr_flags)

        print(f"DONE: {clip.name}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
