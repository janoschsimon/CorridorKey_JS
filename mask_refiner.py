"""
mask_refiner.py — Interactive AlphaHint refiner for CorridorKey.

After MatAnyone2 generates raw masks, this tool lets you tune Dilate and Feather
interactively with a live preview overlaid on the original frame.
Especially useful for hair and fine detail on the edges.

Usage (via GUI, launched automatically after Phase 1):
    MatAnyone2/.venv/Scripts/python.exe mask_refiner.py
        --raw-masks <AlphaHintRaw/>
        --input-frames <Input/>
        --output <AlphaHint/>
        [--dilate N] [--blur N] [--reverse]
"""

import argparse
import os
import threading
import time

import cv2
import gradio as gr
import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

MAX_DISPLAY_W = 1280


def coarsen_mask(mask_u8: np.ndarray, dilate_px: int, blur_px: int) -> np.ndarray:
    if dilate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        mask_u8 = cv2.dilate(mask_u8, kernel)
    if blur_px > 0:
        blur_px = blur_px if blur_px % 2 == 1 else blur_px + 1
        mask_u8 = cv2.GaussianBlur(mask_u8, (blur_px, blur_px), 0)
    return mask_u8


def make_overlay(frame_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Green overlay at 50% max opacity, fading with mask values (soft edges visible)."""
    overlay = frame_rgb.copy().astype(np.float32)
    tint = np.array([50, 220, 50], dtype=np.float32)
    alpha = mask_u8.astype(np.float32) / 255.0 * 0.55
    alpha3 = alpha[:, :, np.newaxis]
    overlay = overlay * (1.0 - alpha3) + tint * alpha3
    return np.clip(overlay, 0, 255).astype(np.uint8)


def load_reference_frame(input_frames_dir: str, reverse: bool) -> np.ndarray:
    """Load first (or last) input frame as RGB uint8."""
    exts = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")
    files = sorted(f for f in os.listdir(input_frames_dir) if f.lower().endswith(exts))
    assert files, f"No frames in {input_frames_dir}"
    target = files[-1] if reverse else files[0]
    img = cv2.imread(os.path.join(input_frames_dir, target), cv2.IMREAD_UNCHANGED)
    assert img is not None, f"Could not read {target}"
    if img.dtype == np.float32:
        img = np.clip(img, 0, 1)
        srgb = np.where(img <= 0.0031308, img * 12.92,
                        1.055 * np.power(np.maximum(img, 1e-10), 1 / 2.4) - 0.055)
        img = (srgb * 255).clip(0, 255).astype(np.uint8)
    return img[..., ::-1].copy()  # BGR → RGB


def main():
    parser = argparse.ArgumentParser(description="CorridorKey interactive mask refiner")
    parser.add_argument("--raw-masks", required=True, help="Folder with raw MatAnyone2 masks")
    parser.add_argument("--input-frames", required=True, help="Input/ folder with original frames")
    parser.add_argument("--output", required=True, help="AlphaHint/ output folder")
    parser.add_argument("--dilate", type=int, default=10)
    parser.add_argument("--blur", type=int, default=5)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    # Load reference frame
    print("[MaskRefiner] Loading reference frame...")
    frame_rgb = load_reference_frame(args.input_frames, args.reverse)
    orig_h, orig_w = frame_rgb.shape[:2]

    scale = min(1.0, MAX_DISPLAY_W / orig_w)
    disp_w, disp_h = int(orig_w * scale), int(orig_h * scale)
    disp_rgb = cv2.resize(frame_rgb, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_rgb.copy()

    # Load raw masks
    mask_files = sorted(f for f in os.listdir(args.raw_masks) if f.lower().endswith(".png"))
    assert mask_files, f"No masks in {args.raw_masks}"
    ref_mask_file = mask_files[-1] if args.reverse else mask_files[0]
    raw_ref_mask = cv2.imread(os.path.join(args.raw_masks, ref_mask_file), cv2.IMREAD_GRAYSCALE)
    assert raw_ref_mask is not None

    # Resize mask to display size for live preview
    if scale < 1.0:
        raw_ref_disp = cv2.resize(raw_ref_mask, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        raw_ref_disp = raw_ref_mask.copy()

    print(f"[MaskRefiner] {len(mask_files)} frames, preview at {disp_w}x{disp_h}")

    done_event = threading.Event()

    def update_preview(dilate, blur):
        coarsened = coarsen_mask(raw_ref_disp.copy(), int(dilate), int(blur))
        return make_overlay(disp_rgb, coarsened)

    def apply_and_run(dilate, blur):
        dilate, blur = int(dilate), int(blur)
        print(f"[MaskRefiner] Applying dilate={dilate} blur={blur} to {len(mask_files)} frames...")
        os.makedirs(args.output, exist_ok=True)
        for fname in mask_files:
            raw = cv2.imread(os.path.join(args.raw_masks, fname), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue
            coarsened = coarsen_mask(raw, dilate, blur)
            cv2.imwrite(os.path.join(args.output, fname), coarsened)
        print(f"[MaskRefiner] Done! Masks saved to {args.output}")
        done_event.set()
        return f"✓ {len(mask_files)} masks saved — CorridorKey starts automatically..."

    with gr.Blocks(title="CorridorKey — Mask Refiner") as demo:
        gr.Markdown("## AlphaHint Refiner")
        gr.Markdown(
            "Tune **Dilate** and **Feather** until the green overlay softly covers the subject.  \n"
            "The fade at the edges is what CorridorKey uses for fine detail like hair.  \n"
            "When happy → **✓ Apply & Run CorridorKey**."
        )

        preview = gr.Image(
            value=update_preview(args.dilate, args.blur),
            label=f"Frame {'(last)' if args.reverse else '0'} — green = AlphaHint region",
            height=disp_h + 60,
            interactive=False,
        )

        with gr.Row():
            dilate_slider = gr.Slider(0, 150, value=args.dilate, step=1, label="Dilate (px) — expand mask outward")
            blur_slider = gr.Slider(1, 101, value=args.blur if args.blur % 2 == 1 else args.blur + 1,
                                    step=2, label="Feather (px) — edge softness / fade")

        run_btn = gr.Button("✓ Apply & Run CorridorKey", variant="primary", size="lg")
        status = gr.Textbox(label="Status", interactive=False, lines=1)

        dilate_slider.change(fn=update_preview, inputs=[dilate_slider, blur_slider], outputs=preview)
        blur_slider.change(fn=update_preview, inputs=[dilate_slider, blur_slider], outputs=preview)
        run_btn.click(fn=apply_and_run, inputs=[dilate_slider, blur_slider], outputs=status)

    print("[MaskRefiner] Starting Gradio — browser will open automatically.")
    demo.launch(inbrowser=True, server_name="127.0.0.1", prevent_thread_lock=True, quiet=True)

    done_event.wait()
    time.sleep(1)
    os._exit(0)


if __name__ == "__main__":
    main()
