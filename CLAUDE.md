# CorridorKey — CLAUDE.md

> **ICH, Claude Code, Flöte vom Dienst, habe dieses Tool NICHT geschrieben und werde KEINE Annahmen machen oder Halluzinationen produzieren!!!!!!!!!!!**

---

## Project
AI-powered greenscreen keyer by Corridor Digital. Neural net separates foreground + alpha from greenscreen footage. Requires a coarse Alpha Hint mask as input.

## Owner
Janosch (Chef). VFX professional, ComfyUI power user, ex-Nuke user. Technical — skip basics, go straight to the point. Prefers to be asked before anything is installed/executed.

## Stack
- Python 3.11.14 via `uv`
- PyTorch 2.10.0 + CUDA 12.6
- GPU: ~24 GB VRAM (RTX 4090 or similar)
- OS: Windows 11, shell: bash

## Running Inference
```bash
# Pipe answers: gamma(s/l), despill(0-10), despeckle(y/n), despeckle size, refiner strength
echo -e "s\n0\ny\n10\n1.0" | uv run python clip_manager.py --action run_inference
```
Clips must live in `ClipsForInference/<shotname>/` with subfolders `Input/` and `AlphaHint/`.

## Folder Structure per Shot
```
ClipsForInference/
└── shotname/
    ├── Input/          ← RGB greenscreen frames
    ├── AlphaHint/      ← coarse B&W mask frames
    └── Output/
        ├── Comp/       ← 8-bit PNG checkerboard preview
        ├── Processed/  ← EXR RGBA linear premult (main deliverable)
        ├── Matte/      ← EXR alpha only
        └── FG/         ← EXR straight foreground (sRGB → convert to linear in comp)
```

## Alpha Hint Generation
- GVM (80 GB VRAM) and VideoMaMa (80 GB) are NOT usable on this machine
- **Current workflow: `generate_alpha_hints.py` — fully automatic, no ComfyUI needed!**
  - Auto chroma key for frame 1 → MatAnyone2 propagates through clip → dilate+blur → AlphaHint
  - Runs under `MatAnyone2/.venv/Scripts/python.exe generate_alpha_hints.py -i <video> -o <AlphaHint/>`
  - MatAnyone2 located at `MatAnyone2/` (subdir), model auto-downloads on first run (135 MB)
- Model expects **coarse/blurry/eroded** masks — NOT sharp/precise
  - Output is dilated (~15px) + gaussian blurred (~31px) for this reason
- Old workflow (kept as fallback): SAM3 in ComfyUI, KEIN GrowMask, GrowMask 0px

## Color Space Rules (DO NOT BREAK)
- Model input/output: `[0.0, 1.0]` float, assumes sRGB input
- FG output: sRGB straight (must convert to linear in comp before premult)
- Alpha output: inherently linear
- Processed EXR: linear premultiplied (use `cu.srgb_to_linear()` — NOT gamma 2.2)
- Inference always at 2048x2048 internally, resized back to source res

## Key Files
- `clip_manager.py` — main CLI, scan/config/inference loop
- `generate_alpha_hints.py` — auto AlphaHint generator (chroma key → MatAnyone2 → coarsen)
- `corridorkey_gui.py` — tkinter GUI frontend
- `CorridorKeyModule/inference_engine.py` — `CorridorKeyEngine`, `process_frame()`
- `CorridorKeyModule/core/model_transformer.py` — GreenFormer architecture
- `CorridorKeyModule/core/color_utils.py` — sRGB/linear math, despill
- `docs/LLM_HANDOVER.md` — detailed architecture notes

## Wizard Mode
`--action wizard` is NOT YET IMPLEMENTED. Use `run_inference` with piped stdin.

## TODO
See `docs/TODO.md`
