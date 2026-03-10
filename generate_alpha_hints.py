"""
generate_alpha_hints.py
-----------------------
Full pipeline: MP4/MOV/EXR-folder/PNG-folder → MatAnyone2 alpha matte → coarse AlphaHint PNG sequence.

Runs under MatAnyone2's venv:
    MatAnyone2/.venv/Scripts/python.exe generate_alpha_hints.py -i <input> -o <alphahint_dir>

Supported inputs:
  - MP4 / MOV / AVI video files  → HSV chroma key, sRGB uint8
  - EXR image sequences           → linear ratio chroma key, exports original EXR to Input/
  - PNG / JPEG image folders      → HSV chroma key, sRGB uint8

Steps:
  1. Auto-detect input type, load frames
  2. Chroma key on reference frame (HSV for sRGB, channel-ratio for linear EXR)
  3. MatAnyone2 inference → per-frame alpha
  4. Dilate + blur → coarse masks CorridorKey expects
  5. Save AlphaHints (+ optional Input/ export, EXR preserved as linear EXR)
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# ── MatAnyone2 imports ────────────────────────────────────────────────────────
MATANYONE2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MatAnyone2")
sys.path.insert(0, MATANYONE2_DIR)

from hugging_face.tools.download_util import load_file_from_url
from matanyone2.inference.inference_core import InferenceCore
from matanyone2.utils.device import get_default_device, safe_autocast_decorator
from matanyone2.utils.get_default_model import get_matanyone2_model
from matanyone2.utils.inference_utils import read_frame_from_videos


# ── SAM person segmentation ───────────────────────────────────────────────────

SAM_CHECKPOINT = os.path.join(MATANYONE2_DIR, "pretrained_models", "sam_vit_h_4b8939.pth")

def crop_letterbox(frame_rgb: np.ndarray, black_thresh: int = 10):
    """Detect and crop black letterbox/pillarbox bars. Returns (cropped, x0, y0)."""
    gray = np.mean(frame_rgb, axis=2)
    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)
    active_cols = np.where(col_means > black_thresh)[0]
    active_rows = np.where(row_means > black_thresh)[0]
    if len(active_cols) == 0 or len(active_rows) == 0:
        return frame_rgb, 0, 0
    x0, x1 = int(active_cols[0]), int(active_cols[-1]) + 1
    y0, y1 = int(active_rows[0]), int(active_rows[-1]) + 1
    cropped = frame_rgb[y0:y1, x0:x1]
    if cropped.shape[:2] != frame_rgb.shape[:2]:
        print(f"[AlphaHint] Letterbox crop: {frame_rgb.shape[1]}x{frame_rgb.shape[0]} -> {cropped.shape[1]}x{cropped.shape[0]} (offset {x0},{y0})")
    return cropped, x0, y0


def generate_sam_mask(frame_rgb: np.ndarray, device: str = "cuda") -> np.ndarray:
    """Grounding DINO finds 'person' bbox, SAM refines to pixel mask.

    1. Crop black letterbox/pillarbox bars
    2. DINO: text prompt 'person' -> bounding box
    3. SAM: bbox prompt -> precise person mask
    4. Place mask back in full-frame coordinates
    """
    from segment_anything import sam_model_registry, SamPredictor
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    from PIL import Image as PILImage

    full_h, full_w = frame_rgb.shape[:2]
    cropped, x0, y0 = crop_letterbox(frame_rgb)
    crop_h, crop_w = cropped.shape[:2]

    # 1. Grounding DINO — find person bbox
    print("[AlphaHint] Grounding DINO: finding 'person'...")
    dino_model_id = "IDEA-Research/grounding-dino-base"
    dino_processor = AutoProcessor.from_pretrained(dino_model_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(device)

    pil_img = PILImage.fromarray(cropped)
    inputs = dino_processor(images=pil_img, text="person", return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.3,
        text_threshold=0.3,
        target_sizes=[(crop_h, crop_w)],
    )[0]

    if len(results["boxes"]) == 0:
        raise RuntimeError("Grounding DINO found no 'person' in reference frame.")

    # Pick highest-confidence person box
    best_idx = results["scores"].argmax().item()
    box = results["boxes"][best_idx].cpu().numpy()  # [x0, y0, x1, y1]
    score = results["scores"][best_idx].item()
    print(f"[AlphaHint] DINO: person found, confidence={score:.2f}, box={box.astype(int).tolist()}")

    del dino_model, dino_processor
    torch.cuda.empty_cache()

    # 2. SAM — refine bbox to pixel mask
    print("[AlphaHint] SAM: refining person mask...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(cropped)

    masks, scores, _ = predictor.predict(
        box=box[None, :],  # SAM expects [1, 4]
        multimask_output=False,
    )
    person_mask = masks[0]  # [H, W] bool
    print(f"[AlphaHint] SAM: mask coverage={person_mask.mean()*100:.1f}%")

    # 3. Place back into full-frame coordinates
    full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
    full_mask[y0:y0 + crop_h, x0:x0 + crop_w] = person_mask.astype(np.uint8) * 255

    del sam, predictor
    torch.cuda.empty_cache()

    return full_mask


# ── Color space ───────────────────────────────────────────────────────────────

def linear_to_srgb_u8(img_linear: np.ndarray) -> np.ndarray:
    """Convert linear float32 BGR [0, inf] to sRGB uint8 BGR [0, 255].

    Highlights above 1.0 are clipped. Used to prepare EXR frames for MatAnyone2.
    """
    img = np.clip(img_linear, 0.0, 1.0)
    srgb = np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0.0031308), 1.0 / 2.4) - 0.055,
    )
    return np.clip(srgb * 255, 0, 255).astype(np.uint8)


# ── Chroma key ────────────────────────────────────────────────────────────────

def generate_chroma_mask(frame_bgr: np.ndarray, hsv_lower=(35, 40, 40), hsv_upper=(90, 255, 255)) -> np.ndarray:
    """HSV chroma key for sRGB uint8 input. Returns 8-bit mask: 255=foreground, 0=green."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(hsv_lower), np.array(hsv_upper))
    mask = cv2.bitwise_not(green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def generate_chroma_mask_linear(frame_linear_bgr: np.ndarray, ratio: float = 1.3, min_green: float = 0.05) -> np.ndarray:
    """Ratio-based chroma key for linear float32 EXR input. Returns 8-bit mask: 255=foreground.

    Works in linear light: green pixels are those where G > R*ratio AND G > B*ratio AND G > min_green.
    More accurate than HSV for linear EXR because no gamma distortion of hue angles.
    """
    b, g, r = frame_linear_bgr[:, :, 0], frame_linear_bgr[:, :, 1], frame_linear_bgr[:, :, 2]
    is_green = (g > r * ratio) & (g > b * ratio) & (g > min_green)
    mask = (~is_green).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


# ── EXR loading ───────────────────────────────────────────────────────────────

def load_exr_sequence(folder: str, max_dim: int = 1280):
    """Load EXR folder, downscaled for MatAnyone2.

    Returns:
        srgb_tensor: [N, C, H, W] uint8 RGB tensor, resized to max_dim longest side
        exr_paths: absolute file paths for lazy export (originals read from disk at save time)
        file_list: sorted filenames

    Downscaling rationale: AlphaHint masks are coarse (dilate+blur), MatAnyone2 doesn't
    need full sensor resolution for tracking. Reduces RAM from ~5 GB to ~300 MB for
    150x2880x1620 → 150x1280x720, and cuts VRAM dramatically.
    Original full-res EXR is preserved on disk and copied to Input/ unmodified.
    """
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".exr"))
    assert files, f"No EXR files found in {folder}"

    # Determine resize dimensions from first frame
    probe = cv2.imread(os.path.join(folder, files[0]), cv2.IMREAD_UNCHANGED)
    assert probe is not None, f"Could not read EXR: {files[0]}"
    h, w = probe.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    if scale < 1.0:
        print(f"[AlphaHint] EXR {w}x{h} -> resized to {new_w}x{new_h} for MatAnyone2 (scale {scale:.2f})")
    else:
        print(f"[AlphaHint] EXR {w}x{h} — no resize needed")

    srgb_tensors = []
    exr_paths = []
    for f in tqdm.tqdm(files, desc="Loading EXR"):
        fpath = os.path.join(folder, f)
        img_bgr = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        assert img_bgr is not None, f"Could not read EXR: {f}"
        if img_bgr.dtype != np.float32:
            img_bgr = img_bgr.astype(np.float32)
        exr_paths.append(fpath)
        img_u8_bgr = linear_to_srgb_u8(img_bgr)
        if scale < 1.0:
            img_u8_bgr = cv2.resize(img_u8_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_u8_rgb = img_u8_bgr[..., ::-1].copy()  # BGR → RGB for MatAnyone2
        srgb_tensors.append(torch.from_numpy(img_u8_rgb).permute(2, 0, 1))  # [C, H, W]
    return torch.stack(srgb_tensors), exr_paths, files


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

EXR_WRITE_FLAGS = [
    cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
    cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PXR24,
]


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

    # 2. Detect input type and load frames
    is_exr = os.path.isdir(args.input) and any(
        f.lower().endswith(".exr") for f in os.listdir(args.input)
    )

    if is_exr:
        print("[AlphaHint] EXR sequence detected — originals preserved.")
        vframes, exr_paths, exr_files = load_exr_sequence(args.input, max_dim=args.matanyone_max_dim)
        total_real_frames = len(exr_files)

        if args.reverse:
            print("[AlphaHint] Reverse mode: last frame as reference.")
            ref_linear = cv2.imread(exr_paths[-1], cv2.IMREAD_UNCHANGED).astype(np.float32)
            vframes = torch.flip(vframes, [0])
            exr_paths = list(reversed(exr_paths))
        else:
            ref_linear = cv2.imread(exr_paths[0], cv2.IMREAD_UNCHANGED).astype(np.float32)

        if args.sam:
            ref_rgb = linear_to_srgb_u8(ref_linear)[..., ::-1].copy()  # BGR->RGB for SAM
            chroma_mask = generate_sam_mask(ref_rgb, device=device)
        else:
            print("[AlphaHint] Chroma key (linear ratio)...")
            chroma_mask = generate_chroma_mask_linear(ref_linear, ratio=args.green_ratio, min_green=args.green_min)

    else:
        exr_paths = None
        vframes, fps, total_real_frames, video_name = read_frame_from_videos(args.input)

        if args.reverse:
            print("[AlphaHint] Reverse mode: last frame as reference.")
            ref_bgr = vframes[-1].permute(1, 2, 0).numpy().astype(np.uint8)[..., ::-1].copy()
            vframes = torch.flip(vframes, [0])
        else:
            ref_bgr = None
            if os.path.isdir(args.input):
                first_frame_file = sorted(os.listdir(args.input))[0]
                ref_bgr = cv2.imread(os.path.join(args.input, first_frame_file))
            else:
                cap = cv2.VideoCapture(args.input)
                ret, ref_bgr = cap.read()
                cap.release()
                assert ret, "Could not read first frame from video."

        if args.sam:
            ref_rgb = ref_bgr[..., ::-1].copy()  # BGR->RGB for SAM
            chroma_mask = generate_sam_mask(ref_rgb, device=device)
        else:
            print("[AlphaHint] Chroma key (HSV)...")
            chroma_mask = generate_chroma_mask(ref_bgr, hsv_lower=tuple(args.hsv_lower), hsv_upper=tuple(args.hsv_upper))

    fg_pct = np.count_nonzero(chroma_mask) / chroma_mask.size * 100
    print(f"[AlphaHint] Foreground coverage on reference frame: {fg_pct:.1f}%")

    # Resize chroma mask to match (possibly downscaled) vframes spatial dimensions
    target_h, target_w = vframes.shape[2], vframes.shape[3]
    if chroma_mask.shape != (target_h, target_w):
        chroma_mask = cv2.resize(chroma_mask, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 3. Warmup padding (stabilizes MatAnyone2 on first frames)
    n_warmup = 10
    repeated = vframes[0].unsqueeze(0).repeat(n_warmup, 1, 1, 1)
    vframes = torch.cat([repeated, vframes], dim=0).float()
    length = total_real_frames + n_warmup

    mask_t = torch.from_numpy(chroma_mask).float().to(device)
    objects = [1]

    # 4. MatAnyone2 inference
    os.makedirs(args.output, exist_ok=True)
    if args.export_input:
        os.makedirs(args.export_input, exist_ok=True)
    print(f"[AlphaHint] Running MatAnyone2 on {total_real_frames} frames...")

    frame_idx = 0
    for ti in tqdm.tqdm(range(length), desc="MatAnyone2"):
        image = vframes[ti]
        image_in = (image / 255.0).float().to(device)

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

        # In reverse mode, map back to original frame index
        save_idx = (total_real_frames - 1 - frame_idx) if args.reverse else frame_idx
        cv2.imwrite(os.path.join(args.output, f"{save_idx:04d}.png"), pha_coarse)

        # 6. Export input frame
        if args.export_input:
            if is_exr and exr_paths is not None:
                # Read original EXR from disk on-demand — no float32 frames kept in RAM
                src_linear = cv2.imread(exr_paths[frame_idx], cv2.IMREAD_UNCHANGED)
                cv2.imwrite(
                    os.path.join(args.export_input, f"{save_idx:04d}.exr"),
                    src_linear,
                    EXR_WRITE_FLAGS,
                )
            else:
                frame_bgr = image.permute(1, 2, 0).numpy().astype(np.uint8)[..., ::-1]
                cv2.imwrite(os.path.join(args.export_input, f"{save_idx:04d}.png"), frame_bgr)

        frame_idx += 1

    print(f"[AlphaHint] Done! {frame_idx} frames saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AlphaHint masks via MatAnyone2")
    parser.add_argument("-i", "--input", required=True, help="Input: video file (mp4/mov/avi), EXR folder, or PNG frame folder")
    parser.add_argument("-o", "--output", required=True, help="Output AlphaHint folder")
    parser.add_argument("--dilate", type=int, default=70, help="Dilate radius (px). Default: 70")
    parser.add_argument("--blur", type=int, default=20, help="Gaussian blur radius (px). Default: 20")
    parser.add_argument("--hsv-lower", type=int, nargs=3, default=[35, 40, 40], metavar=("H", "S", "V"), help="HSV lower bound for sRGB chroma key. Default: 35 40 40")
    parser.add_argument("--hsv-upper", type=int, nargs=3, default=[90, 255, 255], metavar=("H", "S", "V"), help="HSV upper bound for sRGB chroma key. Default: 90 255 255")
    parser.add_argument("--green-ratio", type=float, default=1.3, help="G/R and G/B ratio threshold for linear EXR chroma key. Default: 1.3")
    parser.add_argument("--green-min", type=float, default=0.05, help="Min green value for linear EXR chroma key. Default: 0.05")
    parser.add_argument("--export-input", type=str, default=None, help="Also export input frames here (PNG for video, EXR preserved for EXR input)")
    parser.add_argument("--reverse", action="store_true", help="Process frames reversed, use last frame as chroma key reference")
    parser.add_argument("--sam", action="store_true", help="Use SAM (Segment Anything) instead of chroma key to find subject on reference frame")
    parser.add_argument("--matanyone-max-dim", type=int, default=1280, help="Max longest dimension for MatAnyone2 (EXR only). Default: 1280")
    args = parser.parse_args()
    run(args)
