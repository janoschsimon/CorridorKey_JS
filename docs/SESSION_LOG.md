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
- GUI weiter verfeinern
- Längere Clips testen

---

## 2026-03-09 — Session 1 (Fortsetzung): GUI gebaut & weitere Tests

### Was gemacht wurde
- `run_inference_direct.py` gebaut — bypassed interactive input() Prompts via CLI Args
- `corridorkey_gui.py` gebaut — tkinter GUI mit Ordner-Auswahl, Settings, Progressbar, Log, Kill-on-close
- `CorridorKey_GUI.bat` — Doppelklick-Launcher
- Bug gefixt: stdout-Pipe-Deadlock → stdout in separatem Thread lesen
- Bug gefixt: alle Clips wurden verarbeitet → `--shot` Flag damit nur neuer Clip läuft
- Zweiter Testrun: 135 Frames Actiontest ✅
- Dritter Testrun: 242 Frames ✅ — Speicher wird direkt freigegeben
- Nico von Corridor hat ComfyUI Screenshot mit 🔥 kommentiert
- SAM3 Text-Prompt Workflow bestätigt: "person", "ball", etc. funktioniert perfekt

### Erkenntnisse
- SAM3 mit Text-Prompt ist mächtiger als GVM (hat Kontrolle, braucht kaum VRAM)
- GUI Kill-on-close funktioniert — kein Zombie-Prozess mehr
- Inference braucht ~8 GB VRAM nach Model-Load (23 GB), Speicher nach dem Run direkt frei

### Nächste Schritte
- Garbage Matte Workflow in Resolve testen
- Längere Clips / Red Komodo RAW testen
- GUI weiter polish (z.B. Output-Ordner direkt öffnen nach Fertigstellung)

---

## 2026-03-09 — Session 1 (Teil 3): Mask Workflow Research

### Erkenntnisse (WICHTIG)
- **GrowMask = FALSCH!** Modell erwartet coarse/blurry Masken, GrowMask macht sie zu groß/scharf
- **SAM3 roh (0px Grow)** = bester Workflow für saubere Footage
- **RMBG** keyt schon fertig → nicht geeignet als Hint (zu sauber, Greenscreen weg)
- Optimale Hint: knapp größer als Subjekt, Greenscreen-Rand noch sichtbar
- Bei löchriger SAM3 Maske (Linda RED Komodo): fehlende Körperteile = fehlendes Ergebnis
- run_inference_direct.py hat Bugs → GUI nutzt jetzt wieder clip_manager.py

### Nächste Schritte
- RMBG + kleiner Grow (10-15px) als Alternative zu SAM3 testen
- Linda RED Komodo Workflow lösen (löchrige Maske Problem)
- GUI Mask-Fill Feature einbauen
