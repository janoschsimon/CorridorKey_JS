# Session Log

## 2026-03-09 — Session 1: Installation & Erster Testlauf

### Was gemacht wurde
- `uv sync` ausgeführt → Python 3.11.14, PyTorch 2.10.0+cu126, CUDA verfügbar
- `CorridorKey_v1.0.pth` von HuggingFace geladen → umbenannt zu `CorridorKey.pth` in `CorridorKeyModule/checkpoints/`
- Testclip `2026.03.09_corridortest` vorbereitet:
  - 240 Input-Frames (ComfyUI_candice_XXXXX_.png) — 5D Mark II RAW, sRGB 1920x1080
  - 240 AlphaHint-Frames (ComfyUI_mask_XXXXX_.png) — SAM3 in ComfyUI mit GrowMask +50px
  - In `ClipsForInference/2026.03.09_corridortest/Input/` und `/AlphaHint/` sortiert
- Inference gestartet: `echo -e "s\n0\ny\n10\n1.0" | uv run python clip_manager.py --action run_inference`
- **Ergebnis: Erfolgreich!** Haare sauber freigestellt, Photoshop Slap Comp sieht gut aus

### Erkenntnisse
- GVM und VideoMaMa fallen weg (80 GB VRAM benötigt, GPU hat ~24 GB)
- SAM3 in ComfyUI funktioniert gut als Alpha Hint Generator
- SAM3 Limit: 240 Frames pro Run → für längere Clips Batching nötig
- `--action wizard` ist nicht implementiert → stdin pipen als Workaround
- Comp-Previews (PNG) sind nur grobe Vorschau — Processed EXRs sind das echte Ergebnis

### Nächste Schritte
- Workflow mit Garbage Matte in Resolve vor SAM3 testen
- GUI planen und bauen
- SAM3 Batching für längere Clips lösen
