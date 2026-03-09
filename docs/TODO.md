# CorridorKey — TODO

## High Priority

- [ ] **GUI bauen** — Desktop-App statt Drag & Drop / CLI
  - Clips laden (Datei/Ordner-Dialog)
  - Einstellungen (Gamma, Despill, Despeckle, Refiner)
  - Fortschrittsanzeige (Frame X / Y)
  - Output-Vorschau (Comp-PNG inline)
  - Batch-Verarbeitung mehrerer Shots

- [ ] **Längere Clips testen** — 240 Frames war nur Testwahl, kein echtes Limit
  - Inference braucht nur ~8 GB VRAM → viel mehr Frames möglich
  - SAM3 Limits in ComfyUI noch klären

## Medium Priority

- [ ] **Workflow-Doku** für Janosch's Pipeline
  - Garbage Matte in Resolve → Export → SAM3 in ComfyUI → CorridorKey
  - Red Komodo RAW Pipeline (color space handling)

- [ ] **Alpha Hint Qualität verbessern**
  - GrowMask Wert optimieren (aktuell 50px, testen mit 80-120px)
  - Garbage Matte davor schalten für sauberere Hints

- [ ] **Inference beschleunigen**
  - Aktuell: frame-by-frame, kein Batching auf GPU
  - GPU-Batch-Processing prüfen (mehrere Frames parallel wenn VRAM reicht)

## Nice to Have

- [ ] Resolve-Integration (OTIO, EDL, oder DaVinci Scripting API)
- [ ] Direkter ComfyUI-Node für CorridorKey
- [ ] Automatisches Hint-Generieren mit leichteren Alternativen zu GVM/VideoMaMa

## Done ✅

- [x] Dependencies installiert (PyTorch 2.10 + CUDA 12.6)
- [x] CorridorKey.pth Model geladen (383 MB)
- [x] Erster Testlauf erfolgreich — 240 Frames Candice Greenscreen
- [x] SAM3 als Alpha Hint Workflow etabliert
- [x] GUI gebaut (corridorkey_gui.py + CorridorKey_GUI.bat)
- [x] run_inference_direct.py — bypasses interactive prompts, CLI args
- [x] Kill-on-close implementiert in GUI
