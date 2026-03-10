"""
CorridorKey GUI
Simple tkinter frontend for running CorridorKey inference.
Usage: uv run python corridorkey_gui.py
"""

import os
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

# Must be set before cv2 is imported anywhere in this process
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "ClipsForInference")
MATANYONE2_PYTHON = os.path.join(BASE_DIR, "MatAnyone2", ".venv", "Scripts", "python.exe")
GENERATE_HINTS_SCRIPT = os.path.join(BASE_DIR, "generate_alpha_hints.py")
MASK_PAINTER_SCRIPT = os.path.join(BASE_DIR, "mask_painter.py")
MASK_REFINER_SCRIPT = os.path.join(BASE_DIR, "mask_refiner.py")

VIDEO_EXTS = (".mp4", ".mov", ".avi")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))


class CorridorKeyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CorridorKey")
        self.root.resizable(False, False)
        self._proc = None
        self._input_type = "video"  # "video" | "exr" | "manual_shot"
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # --- Input ---
        input_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        input_frame.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self.input_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_var, width=46).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(input_frame, text="MP4 / MOV", command=self._browse_video).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(input_frame, text="Frame Folder", command=self._browse_folder).grid(row=0, column=2)

        self.info_label = ttk.Label(input_frame, text="", foreground="gray")
        self.info_label.grid(row=1, column=0, columnspan=3, sticky="w", pady=(5, 0))

        # --- Shot Name ---
        name_frame = ttk.LabelFrame(self.root, text="Shot Name", padding=10)
        name_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        self.shot_name_var = tk.StringVar(value="shot")
        ttk.Entry(name_frame, textvariable=self.shot_name_var, width=40).grid(row=0, column=0, sticky="w")
        ttk.Label(name_frame, text="→ YYYY.MM.DD_<name>", foreground="gray").grid(row=0, column=1, padx=10)

        # --- Alpha Hint ---
        hint_frame = ttk.LabelFrame(self.root, text="Alpha Hint Generation (MatAnyone2)", padding=10)
        hint_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(hint_frame, text="Dilate (px):").grid(row=0, column=0, sticky="w")
        self.dilate_var = tk.IntVar(value=10)
        ttk.Spinbox(hint_frame, from_=0, to=100, textvariable=self.dilate_var, width=5).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(hint_frame, text="Blur (px):").grid(row=0, column=2, sticky="w")
        self.blur_var = tk.IntVar(value=5)
        ttk.Spinbox(hint_frame, from_=0, to=200, textvariable=self.blur_var, width=5).grid(row=0, column=3, sticky="w", padx=(5, 0))

        self.reverse_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(hint_frame, text="Reverse (person on last frame)", variable=self.reverse_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        self.sam_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(hint_frame, text="SAM (no greenscreen)", variable=self.sam_var).grid(row=1, column=2, columnspan=2, sticky="w", pady=(5, 0))

        self.hint_info = ttk.Label(hint_frame, text="Auto-generated from MP4 via chroma key + MatAnyone2", foreground="gray")
        self.hint_info.grid(row=2, column=0, columnspan=4, sticky="w")

        # --- Settings ---
        settings_frame = ttk.LabelFrame(self.root, text="CorridorKey Settings", padding=10)
        settings_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        ttk.Label(settings_frame, text="Gamma:").grid(row=0, column=0, sticky="w")
        self.gamma_var = tk.StringVar(value="sRGB")
        ttk.Combobox(settings_frame, textvariable=self.gamma_var, values=["sRGB", "Linear"], state="readonly", width=10).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(settings_frame, text="Despill (0–10):").grid(row=0, column=2, sticky="w")
        self.despill_var = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=0, to=10, textvariable=self.despill_var, width=5).grid(row=0, column=3, sticky="w", padx=(5, 20))

        ttk.Label(settings_frame, text="Auto-Despeckle:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.despeckle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.despeckle_var, command=self._toggle_despeckle).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(settings_frame, text="Min Size (px):").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.despeckle_size_var = tk.IntVar(value=400)
        self.despeckle_size_spin = ttk.Spinbox(settings_frame, from_=0, to=9999, textvariable=self.despeckle_size_var, width=6)
        self.despeckle_size_spin.grid(row=1, column=3, sticky="w", padx=(5, 20), pady=(8, 0))

        ttk.Label(settings_frame, text="Refiner Strength:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.refiner_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(settings_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.refiner_var, width=5, format="%.1f").grid(row=2, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        # --- Progress ---
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=480)
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        self.progress_label = ttk.Label(progress_frame, text="Ready", foreground="gray")
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(5, 0))

        # --- Buttons ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=15)

        self.start_btn = ttk.Button(btn_frame, text="▶  Start", command=self._start)
        self.start_btn.grid(row=0, column=0, ipadx=20, ipady=5, padx=5)

        self.paint_btn = ttk.Button(btn_frame, text="Paint Mask", command=self._paint_mask)
        self.paint_btn.grid(row=0, column=1, ipadx=10, ipady=5, padx=5)

        self.open_btn = ttk.Button(btn_frame, text="Open Output Folder", command=self._open_output, state="disabled")
        self.open_btn.grid(row=0, column=2, ipadx=10, ipady=5, padx=5)

        self._last_output_path = None

    def _toggle_despeckle(self):
        state = "normal" if self.despeckle_var.get() else "disabled"
        self.despeckle_size_spin.config(state=state)

    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")]
        )
        if not path:
            return
        self.input_var.set(path)
        self._input_type = "video"
        self.hint_info.config(text="Auto-generated from MP4 via chroma key + MatAnyone2", foreground="gray")
        self.info_label.config(text="Analyzing...", foreground="gray")
        threading.Thread(target=self._analyze_video, args=(path,), daemon=True).start()

    def _analyze_video(self, path):
        try:
            import cv2 as _cv2
            cap = _cv2.VideoCapture(path)
            frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(_cv2.CAP_PROP_FPS)
            cap.release()
            info = f"{os.path.basename(path)} — {w}x{h}, {frames} frames @ {fps:.2f} fps, sRGB"
            self.root.after(0, lambda: self.info_label.config(text=f"Video: {info}", foreground="green"))
        except Exception as e:
            self.root.after(0, lambda: self.info_label.config(text=f"Video selected (analysis failed: {e})", foreground="green"))

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select footage folder (EXR sequence, PNG frames, or shot folder)")
        if not folder:
            return
        self.input_var.set(folder)
        self.info_label.config(text="Analyzing...", foreground="gray")
        threading.Thread(target=self._analyze_folder, args=(folder,), daemon=True).start()

    def _analyze_folder(self, folder):
        try:
            import cv2 as _cv2
            files = sorted(os.listdir(folder))
            exr_files = [f for f in files if f.lower().endswith(".exr")]

            if exr_files:
                img = _cv2.imread(os.path.join(folder, exr_files[0]), _cv2.IMREAD_UNCHANGED)
                h, w = img.shape[:2] if img is not None else (0, 0)
                info = f"EXR sequence: {len(exr_files)} frames, {w}x{h}, Linear float32"
                self.root.after(0, lambda: [
                    self.info_label.config(text=f"EXR: {info} — AlphaHint auto-generated", foreground="green"),
                    self.hint_info.config(text="EXR mode: linear ratio chroma key, Input/ exported as linear EXR", foreground="gray"),
                    self.gamma_var.set("Linear"),
                ])
                self._input_type = "exr"
                return

            if os.path.isdir(os.path.join(folder, "Input")):
                input_files = [f for f in os.listdir(os.path.join(folder, "Input")) if is_image_file(f)]
                self.root.after(0, lambda: [
                    self.info_label.config(text=f"Shot folder: {len(input_files)} Input frames (manual mode)", foreground="green"),
                    self.hint_info.config(text="Manual mode: AlphaHint/ must exist alongside Input/", foreground="orange"),
                ])
                self._input_type = "manual_shot"
                return

            img_files = [f for f in files if is_image_file(f)]
            if img_files:
                self.root.after(0, lambda: self.info_label.config(
                    text=f"{len(img_files)} image frames — manual AlphaHint mode", foreground="orange"
                ))
                self._input_type = "manual_shot"
            else:
                self.root.after(0, lambda: self.info_label.config(text="No recognized frames found", foreground="red"))
        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: self.info_label.config(text=f"Analysis error: {err}", foreground="red"))

    def _start(self):
        input_path = self.input_var.get().strip()
        shot_name = self.shot_name_var.get().strip() or "shot"

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid video file or frame folder.")
            return

        self.start_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self._last_output_path = None
        self.progress_label.config(text="Starting...", foreground="blue")
        self.root.update()

        is_video = input_path.lower().endswith(VIDEO_EXTS) or self._input_type == "exr"
        threading.Thread(target=self._run, args=(input_path, shot_name, is_video), daemon=True).start()

    def _run(self, input_path, shot_name, is_video):
        try:
            date_str = datetime.now().strftime("%Y.%m.%d")
            full_shot_name = f"{date_str}_{shot_name}"
            shot_path = os.path.join(CLIPS_DIR, full_shot_name)
            input_dir = os.path.join(shot_path, "Input")
            alpha_dir = os.path.join(shot_path, "AlphaHint")
            comp_dir = os.path.join(shot_path, "Output", "Comp")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(alpha_dir, exist_ok=True)

            import time

            if is_video:
                # Phase 1: AlphaHint generation + frame export via MatAnyone2
                self._update_progress(0, 100, "Step 1/2 — Generating AlphaHint via MatAnyone2...", "blue")
                print(f"[CorridorKey] Input: {os.path.basename(input_path)}")
                print("[CorridorKey] Launching MatAnyone2...")

                raw_dir = os.path.join(shot_path, "AlphaHintRaw")
                cmd = [
                    MATANYONE2_PYTHON, GENERATE_HINTS_SCRIPT,
                    "-i", input_path,
                    "-o", alpha_dir,
                    "--export-input", input_dir,
                    "--raw-output", raw_dir,
                    "--dilate", str(self.dilate_var.get()),
                    "--blur", str(self.blur_var.get()),
                ]
                if self.reverse_var.get():
                    cmd.append("--reverse")
                if self.sam_var.get():
                    cmd.append("--sam")

                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                # stdout/stderr go directly to CMD window
                proc = subprocess.Popen(cmd, env=env)
                self._proc = proc

                while proc.poll() is None:
                    done = len([f for f in os.listdir(alpha_dir) if is_image_file(f)]) if os.path.exists(alpha_dir) else 0
                    total = len([f for f in os.listdir(input_dir) if is_image_file(f)]) if os.path.exists(input_dir) else 1
                    pct = min(done / max(total, 1) * 50, 50)
                    self._update_progress(pct, 100, f"Step 1/2 — AlphaHint: {done} frames...", "blue")
                    time.sleep(0.5)

                if proc.returncode != 0:
                    raise RuntimeError("AlphaHint generation failed — check CMD window.")

                print("[CorridorKey] AlphaHint generation done.")
                self._run_refiner(raw_dir, input_dir, alpha_dir)
            else:
                # Folder mode: Input/ and AlphaHint/ must exist in the selected folder
                import shutil
                src_input = os.path.join(input_path, "Input")
                src_alpha = os.path.join(input_path, "AlphaHint")
                if os.path.isdir(src_input) and os.path.isdir(src_alpha):
                    for f in os.listdir(src_input):
                        if is_image_file(f):
                            shutil.copy2(os.path.join(src_input, f), input_dir)
                    for f in os.listdir(src_alpha):
                        if is_image_file(f):
                            shutil.copy2(os.path.join(src_alpha, f), alpha_dir)
                else:
                    for f in sorted(os.listdir(input_path)):
                        if is_image_file(f):
                            shutil.copy2(os.path.join(input_path, f), input_dir)

            self._run_phase2(full_shot_name, shot_path, input_dir, comp_dir)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self._update_progress(0, 100, "Failed — check CMD window", "red")
        finally:
            self._proc = None
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

    def _run_refiner(self, raw_dir, input_dir, alpha_dir):
        """Launch interactive mask refiner — blocks until user confirms."""
        if not os.path.isdir(raw_dir) or not os.listdir(raw_dir):
            print("[CorridorKey] No raw masks found — skipping refiner.")
            return
        self._update_progress(50, 100, "Mask Refiner — adjust in browser, then click Apply...", "blue")
        print("[CorridorKey] Launching Mask Refiner in browser...")
        cmd = [
            MATANYONE2_PYTHON, MASK_REFINER_SCRIPT,
            "--raw-masks", raw_dir,
            "--input-frames", input_dir,
            "--output", alpha_dir,
            "--dilate", str(self.dilate_var.get()),
            "--blur", str(self.blur_var.get()),
        ]
        if self.reverse_var.get():
            cmd.append("--reverse")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        proc = subprocess.Popen(cmd, env=env)
        self._proc = proc
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError("Mask refiner failed — check CMD window.")
        print("[CorridorKey] Mask refiner done.")

    def _run_phase2(self, full_shot_name, shot_path, input_dir, comp_dir):
        """Phase 2: CorridorKey inference (shared by auto and paint-mask flows)."""
        total_frames = len([f for f in os.listdir(input_dir) if is_image_file(f)])
        self._update_progress(50, 100, f"Step 2/2 — Running CorridorKey ({total_frames} frames)...", "blue")
        print(f"[CorridorKey] Starting inference on {total_frames} frames...")

        from clip_manager import InferenceSettings, run_inference, scan_clips

        settings = InferenceSettings(
            input_is_linear=(self.gamma_var.get() == "Linear"),
            despill_strength=self.despill_var.get() / 10.0,
            auto_despeckle=self.despeckle_var.get(),
            despeckle_size=self.despeckle_size_var.get(),
            refiner_scale=self.refiner_var.get(),
        )

        clips = scan_clips()
        clips = [c for c in clips if c.name == full_shot_name]
        if not clips:
            raise RuntimeError(f"Shot '{full_shot_name}' not found in ClipsForInference/.")

        def on_frame_complete(i, total):
            pct = 50 + min((i + 1) / max(total, 1) * 50, 50)
            self._update_progress(pct, 100, f"Step 2/2 — CorridorKey: {i + 1} / {total} frames...", "blue")

        run_inference(clips, settings=settings, on_frame_complete=on_frame_complete)

        done = len([f for f in os.listdir(comp_dir) if is_image_file(f)]) if os.path.exists(comp_dir) else total_frames
        self._update_progress(100, 100, f"Done! {done} frames → {full_shot_name}", "green")
        print(f"[CorridorKey] Finished! Output: {shot_path}")
        self._last_output_path = shot_path
        self.root.after(0, lambda: self.open_btn.config(state="normal"))

    def _paint_mask(self):
        input_path = self.input_var.get().strip()
        shot_name = self.shot_name_var.get().strip() or "shot"

        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Error", "Please select a valid video file or frame folder.")
            return
        if self._input_type == "manual_shot":
            messagebox.showerror("Error", "Paint Mask requires a video or EXR folder as input, not a shot folder.")
            return

        self.start_btn.config(state="disabled")
        self.paint_btn.config(state="disabled")
        self.open_btn.config(state="disabled")
        self._last_output_path = None
        threading.Thread(target=self._run_paint_mask, args=(input_path, shot_name), daemon=True).start()

    def _run_paint_mask(self, input_path, shot_name):
        import time
        try:
            date_str = datetime.now().strftime("%Y.%m.%d")
            full_shot_name = f"{date_str}_{shot_name}"
            shot_path = os.path.join(CLIPS_DIR, full_shot_name)
            input_dir = os.path.join(shot_path, "Input")
            alpha_dir = os.path.join(shot_path, "AlphaHint")
            comp_dir = os.path.join(shot_path, "Output", "Comp")
            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(alpha_dir, exist_ok=True)

            self._update_progress(0, 100, "Paint Mask — browser opening...", "blue")
            print("[CorridorKey] Launching mask painter in browser...")

            raw_dir = os.path.join(shot_path, "AlphaHintRaw")
            cmd = [
                MATANYONE2_PYTHON, MASK_PAINTER_SCRIPT,
                "-i", input_path,
                "-o", alpha_dir,
                "--export-input", input_dir,
                "--raw-output", raw_dir,
                "--dilate", str(self.dilate_var.get()),
                "--blur", str(self.blur_var.get()),
            ]
            if self.reverse_var.get():
                cmd.append("--reverse")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # stdout/stderr go directly to CMD window
            proc = subprocess.Popen(cmd, env=env)
            self._proc = proc

            while proc.poll() is None:
                done = len([f for f in os.listdir(alpha_dir) if is_image_file(f)]) if os.path.exists(alpha_dir) else 0
                total = max(done, 1)
                pct = min(done / total * 50, 49)
                self._update_progress(pct, 100, f"Step 1/2 — Painting/Propagating: {done} frames...", "blue")
                time.sleep(0.5)

            if proc.returncode != 0:
                raise RuntimeError("Mask painter / MatAnyone2 failed — check CMD window.")

            print("[CorridorKey] Mask painting done.")
            self._run_refiner(raw_dir, input_dir, alpha_dir)
            self._run_phase2(full_shot_name, shot_path, input_dir, comp_dir)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self._update_progress(0, 100, "Failed — check CMD window", "red")
        finally:
            self._proc = None
            self.root.after(0, lambda: [
                self.start_btn.config(state="normal"),
                self.paint_btn.config(state="normal"),
            ])

    def _open_output(self):
        if self._last_output_path and os.path.isdir(self._last_output_path):
            os.startfile(self._last_output_path)

    def _update_progress(self, value, maximum, message, color="blue"):
        self.root.after(0, lambda: self.progress_var.set(value / maximum * 100))
        self.root.after(0, lambda: self.progress_label.config(text=message, foreground=color))

    def _on_close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CorridorKeyGUI(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()
