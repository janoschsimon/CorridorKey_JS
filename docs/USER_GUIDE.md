# CorridorKey — User Guide

## Starten

Doppelklick auf **`CorridorKey_GUI.bat`** im Projektordner.
Das öffnet die GUI (tkinter-Fenster).

---

## Normaler Workflow (MP4 / MOV mit Greenscreen)

1. **Input** → `MP4 / MOV` klicken, Videodatei auswählen
   → Die GUI zeigt Auflösung, Frameanzahl und FPS an

2. **Shot Name** — Namen vergeben (wird zu `YYYY.MM.DD_<name>`)

3. **Alpha Hint Generation** — Einstellungen meist so lassen:
   - Dilate: 70 px, Blur: 20 px (grob genug für das Modell)
   - **Reverse** — ankreuzen wenn die Person erst gegen Ende ins Bild kommt
     (letzter Frame wird als Referenz für den Chroma-Key genutzt)
   - **SAM** — nur wenn **kein Greenscreen** vorhanden (→ semantische Personenerkennung)

4. **▶ Start** klicken
   - Phase 1: MatAnyone2 generiert AlphaHint-Masken aus dem Chroma-Key
   - Phase 2: CorridorKey rechnet RGBA EXR aus
   - Fortschrittsbalken 0 → 50% = Phase 1, 50 → 100% = Phase 2

5. **Open Output Folder** erscheint wenn fertig
   Haupt-Deliverable: `Output/Processed/` — lineare EXR RGBA premult

---

## Wenn der Auto-Chroma-Key versagt → Paint Mask

Typische Fälle wo der automatische Chroma-Key scheitert:
- Schatten der Person fällt auf das Greenscreen (dunkles Grün → als Vordergrund erkannt)
- Portable Greenscreen mit ungleichmäßiger Beleuchtung
- Kein Greenscreen (→ SAM-Checkbox nutzen)

**Lösung: Paint Mask Button**

1. Video auswählen (wie normal)
2. **Paint Mask** klicken (statt ▶ Start)
   → Browser öffnet sich (Gradio UI)
3. Im Browser: Person auf Frame 0 **grob einmalen** (kein Perfektionismus nötig!)
   - Nur die Person einfärben, **nicht** Schatten oder andere Bereiche
   - Grobe Pinselstriche reichen — das Modell erwartet keine präzise Maske
4. Bestätigen / Submit im Browser
5. MatAnyone2 propagiert die gemalte Maske durch den gesamten Clip
6. Phase 2 startet automatisch

> **Tipp:** Wenn Reverse aktiviert ist, wird der **letzte Frame** zum Malen geöffnet.

---

## EXR-Sequenz als Input

1. **Frame Folder** klicken, EXR-Ordner auswählen
   → GUI erkennt EXR automatisch, setzt Gamma auf "Linear"
2. Workflow identisch wie MP4 (▶ Start oder Paint Mask)
3. Input/ wird als original-lineare EXR exportiert (kein Qualitätsverlust)
4. MatAnyone2 arbeitet intern auf downgeskaltem sRGB (1280px), Output-EXR bleibt volle Auflösung

---

## CorridorKey Einstellungen

| Setting | Default | Erklärung |
|---|---|---|
| Gamma | sRGB | sRGB für MP4/MOV, Linear für EXR (automatisch gesetzt) |
| Despill | 5 | Grünstich entfernen, 0 = aus, 10 = aggressiv |
| Auto-Despeckle | an | Kleine Löcher/Punkte in der Maske entfernen |
| Min Size (px) | 400 | Flecken kleiner als X Pixel werden entfernt |
| Refiner Strength | 1.0 | Matten-Qualität, 0 = schnell/grob, 1–2 = fein/langsam |

---

## Output-Struktur

```
ClipsForInference/YYYY.MM.DD_<shotname>/
├── Input/           ← RGB-Frames (PNG oder EXR)
├── AlphaHint/       ← generierte Rohmasken (PNG)
└── Output/
    ├── Processed/   ← HAUPT-DELIVERABLE: EXR RGBA linear premult
    ├── Matte/       ← EXR Alpha-Kanal einzeln
    ├── FG/          ← EXR Straight FG (sRGB → in Nuke/AE zu linear konvertieren!)
    └── Comp/        ← 8-bit PNG Vorschau (Schachbrettmuster)
```

> **FG/ Hinweis:** Die FG-EXRs sind in sRGB gespeichert.
> In Nuke/AE **vor dem Premult** mit einem Colorspace-Node zu linear konvertieren!

---

## Troubleshooting

| Problem | Ursache | Lösung |
|---|---|---|
| Maske trackt Schatten mit | Schatten dunkelt Grün ab → Chroma-Key erkennt Schatten als FG | **Paint Mask** nutzen, nur Person klicken |
| Portabler Greenscreen wird transparent | CorridorKey keyt alles Grüne — auch das Panel | **Paint Mask** → nur Person klicken (kein Panel) |
| Maske driftet weg von Person | Person bewegt sich stark, MatAnyone2 verliert Tracking | **Reverse** versuchen oder **Paint Mask** |
| Grünstich im Motiv | Despill zu niedrig | Despill auf 7–10 erhöhen |
| Löcher in Haaren/Kanten | AlphaHint zu eng | Dilate erhöhen (z.B. 30–50 px) |
| Sehr langsam | Refiner Strength hoch + viele Frames | Refiner auf 0.5 reduzieren für Test |
| OOM in Phase 1 nach Paint Mask | SAM noch im VRAM wenn MatAnyone2 startet | Bereits gefixt — SAM wird automatisch entladen |

---

## Technischer Hintergrund: Warum Paint Mask für portablen Greenscreen?

CorridorKey bekommt RGB + AlphaHint als 4-Kanal-Input ins neuronale Netz.
Die AlphaHint ist **keine harte Grenze** — das Modell kann auch außerhalb keyen wenn es grüne Pixel sieht.

Bei portablem Greenscreen (ovales Panel): Das Modell sieht das grüne Panel im RGB und keyt es,
auch wenn die AlphaHint nur die Person zeigt.

**Fix (in `inference_engine.py`):** Alpha wird nach dem Netz-Output mit der AlphaHint multipliziert
→ harter Clip → nichts außerhalb der Hint-Region kann transparent werden.

Deshalb: AlphaHint möglichst eng um die Person (Paint Mask + SAM), kleines Dilate (10px).
Nicht das Panel in die Maske einschließen.
