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
  - MatAnyone2 dilate/blur Parameter auf echtem Material optimieren
  - Garbage Matte in Resolve davor schalten für sauberere Frame-1-Maske

- [ ] **MatAnyone2 in GUI integrieren**
  - `generate_alpha_hints.py` unter der Haube aufrufen
  - Input: MP4 → AlphaHint automatisch generieren, dann direkt CorridorKey starten

- [ ] **Inference beschleunigen**
  - Aktuell: frame-by-frame, kein Batching auf GPU
  - GPU-Batch-Processing prüfen (mehrere Frames parallel wenn VRAM reicht)

## Nice to Have

- [ ] Resolve-Integration (OTIO, EDL, oder DaVinci Scripting API)
- [ ] Direkter ComfyUI-Node für CorridorKey

## Done ✅

- [x] Dependencies installiert (PyTorch 2.10 + CUDA 12.8)
- [x] CorridorKey.pth Model geladen (383 MB)
- [x] Erster Testlauf erfolgreich — 240 Frames Candice Greenscreen
- [x] SAM3 als Alpha Hint Workflow etabliert (veraltet, ersetzt durch MatAnyone2)
- [x] GUI gebaut (corridorkey_gui.py + CorridorKey_GUI.bat)
- [x] run_inference_direct.py — bypasses interactive prompts, CLI args
- [x] Kill-on-close implementiert in GUI
- [x] MatAnyone2 als vollautomatischer AlphaHint-Generator (generate_alpha_hints.py)
