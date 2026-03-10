"""
mask_painter.py — Click-to-segment mask creator for CorridorKey.

Click on the person in the reference frame → SAM generates a mask automatically.
Multiple clicks refine the mask. Then MatAnyone2 propagates through the clip
and the result gets dilated+blurred into the coarse AlphaHint CorridorKey expects.

Usage (via GUI "Paint Mask" button, or directly):
    MatAnyone2/.venv/Scripts/python.exe mask_painter.py
        -i <video_or_exr_folder>
        -o <alphahint_output_dir>
        --export-input <input_frames_dir>
        [--dilate N] [--blur N] [--reverse]
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import gradio as gr
import numpy as np
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATE_HINTS_SCRIPT = os.path.join(BASE_DIR, "generate_alpha_hints.py")
SAM_CHECKPOINT = os.path.join(BASE_DIR, "MatAnyone2", "pretrained_models", "sam_vit_h_4b8939.pth")
MAX_DISPLAY_W = 1280


def _linear_to_srgb_u8(img_bgr_f32: np.ndarray) -> np.ndarray:
    img = np.clip(img_bgr_f32, 0.0, 1.0)
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 1e-10), 1.0 / 2.4) - 0.055,
    )
    return (srgb * 255).clip(0, 255).astype(np.uint8)


def extract_reference_frame(input_path: str, reverse: bool = False) -> np.ndarray:
    """Return RGB uint8 HxWx3 reference frame (first or last)."""
    if os.path.isdir(input_path):
        exts = (".exr", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
        files = sorted(f for f in os.listdir(input_path) if f.lower().endswith(exts))
        assert files, f"No frames found in {input_path}"
        target = files[-1] if reverse else files[0]
        img = cv2.imread(os.path.join(input_path, target), cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Could not read: {target}"
        if img.dtype == np.float32:
            img = _linear_to_srgb_u8(img)
        return img[..., ::-1].copy()  # BGR → RGB
    else:
        cap = cv2.VideoCapture(input_path)
        if reverse:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 1))
        ret, frame = cap.read()
        cap.release()
        assert ret, "Could not read frame from video"
        return frame[..., ::-1].copy()  # BGR → RGB


def make_overlay(base_rgb: np.ndarray, mask_bool: np.ndarray | None, points: list) -> np.ndarray:
    """Render blue mask overlay + click dots on top of the base image."""
    overlay = base_rgb.copy().astype(np.float32)
    if mask_bool is not None:
        tint = np.array([50, 120, 255], dtype=np.float32)
        overlay[mask_bool] = overlay[mask_bool] * 0.45 + tint * 0.55
    out = np.clip(overlay, 0, 255).astype(np.uint8)
    for (x, y, label) in points:
        color = (0, 220, 0) if label == 1 else (220, 0, 0)  # green=fg, red=bg
        cv2.circle(out, (int(x), int(y)), 7, color, -1)
        cv2.circle(out, (int(x), int(y)), 7, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="CorridorKey click-to-segment mask creator")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--export-input", default=None)
    parser.add_argument("--dilate", type=int, default=10)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--raw-output", default=None)
    args = parser.parse_args()

    print("[MaskPainter] Loading reference frame...")
    frame_rgb = extract_reference_frame(args.input, reverse=args.reverse)
    orig_h, orig_w = frame_rgb.shape[:2]

    # Downscale for browser display
    scale = min(1.0, MAX_DISPLAY_W / orig_w)
    disp_w = int(orig_w * scale)
    disp_h = int(orig_h * scale)
    display_rgb = (
        cv2.resize(frame_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else frame_rgb.copy()
    )

    # Load SAM once — stays in memory for all clicks, freed before MatAnyone2 runs
    print("[MaskPainter] Loading SAM vit_h (first time: ~5s + model download if needed)...")
    from segment_anything import sam_model_registry, SamPredictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _sam = [sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)]
    _sam[0].to(device)
    _predictor = [SamPredictor(_sam[0])]
    _predictor[0].set_image(display_rgb)
    print(f"[MaskPainter] SAM ready on {device}.")

    done_event = threading.Event()
    current_mask = [None]  # shared: bool ndarray H×W at display resolution, or None

    def on_click(evt: gr.SelectData, points_state):
        x, y = int(evt.index[0]), int(evt.index[1])
        new_points = points_state + [(x, y, 1)]  # 1 = foreground click
        coords = np.array([(p[0], p[1]) for p in new_points])
        labels = np.array([p[2] for p in new_points])
        masks, scores, _ = _predictor[0].predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=True,
        )
        best = masks[scores.argmax()]  # bool H×W
        current_mask[0] = best
        overlay = make_overlay(display_rgb, best, new_points)
        return overlay, new_points

    def on_clear(_points_state):
        current_mask[0] = None
        return display_rgb.copy(), []

    def run_matanyone(points_state):
        if current_mask[0] is None:
            yield "No mask yet — click on the person first."
            return

        # Scale mask back to original resolution
        mask_u8 = current_mask[0].astype(np.uint8) * 255
        if (mask_u8.shape[0], mask_u8.shape[1]) != (orig_h, orig_w):
            mask_u8 = cv2.resize(mask_u8, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        cv2.imwrite(tmp.name, mask_u8)
        print(f"[MaskPainter] Mask saved, coverage={mask_u8.mean() / 255 * 100:.1f}%")

        cmd = [
            sys.executable, GENERATE_HINTS_SCRIPT,
            "-i", args.input,
            "-o", args.output,
            "--mask", tmp.name,
            "--dilate", str(args.dilate),
            "--blur", str(args.blur),
        ]
        if args.export_input:
            cmd += ["--export-input", args.export_input]
        if args.raw_output:
            cmd += ["--raw-output", args.raw_output]
        if args.reverse:
            cmd.append("--reverse")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        # Free SAM from VRAM before MatAnyone2 starts
        _sam[0] = None
        _predictor[0] = None
        torch.cuda.empty_cache()
        print("[MaskPainter] SAM unloaded from VRAM.")

        log = ""
        yield "Starting MatAnyone2...\n"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )

        for line in proc.stdout:
            log += line
            yield log

        proc.wait()
        os.unlink(tmp.name)

        if proc.returncode == 0:
            yield log + "\n✓ Done! You can close this tab — CorridorKey starts automatically."
            done_event.set()
        else:
            yield log + "\n✗ MatAnyone2 failed (see log above)."

    short_name = os.path.basename(args.input.rstrip("/\\"))
    reverse_note = "  *(reverse mode — last frame shown)*" if args.reverse else ""

    with gr.Blocks(title="CorridorKey — Click to Mask") as demo:
        gr.Markdown("## CorridorKey — Click on the person")
        gr.Markdown(
            f"**Input:** `{short_name}`{reverse_note}  \n"
            f"**Left-click** on the person → SAM segments automatically. "
            f"Click multiple times to refine. Green dots = your clicks.  \n"
            f"When happy with the mask, hit **▶ Run MatAnyone2**."
        )

        points_state = gr.State([])

        img_display = gr.Image(
            value=display_rgb.copy(),
            label=f"Frame {'(last)' if args.reverse else '0'} — click the person",
            height=disp_h + 60,
            interactive=False,
        )

        with gr.Row():
            clear_btn = gr.Button("Clear Clicks", size="sm")
            run_btn = gr.Button("▶  Run MatAnyone2", variant="primary", size="lg")

        log_box = gr.Textbox(label="Progress", lines=12, max_lines=20, interactive=False)

        img_display.select(fn=on_click, inputs=points_state, outputs=[img_display, points_state])
        clear_btn.click(fn=on_clear, inputs=points_state, outputs=[img_display, points_state])
        run_btn.click(fn=run_matanyone, inputs=points_state, outputs=log_box)

    print("[MaskPainter] Starting Gradio — browser will open automatically.")
    demo.launch(inbrowser=True, server_name="127.0.0.1", prevent_thread_lock=True, quiet=True)

    done_event.wait()
    time.sleep(2)
    os._exit(0)


if __name__ == "__main__":
    main()
