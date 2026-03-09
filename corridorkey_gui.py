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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "ClipsForInference")
MATANYONE2_PYTHON = os.path.join(BASE_DIR, "MatAnyone2", ".venv", "Scripts", "python.exe")
GENERATE_HINTS_SCRIPT = os.path.join(BASE_DIR, "generate_alpha_hints.py")

VIDEO_EXTS = (".mp4", ".mov", ".avi")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))


class CorridorKeyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CorridorKey")
        self.root.resizable(False, False)
        self._proc = None
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
        self.dilate_var = tk.IntVar(value=70)
        ttk.Spinbox(hint_frame, from_=0, to=100, textvariable=self.dilate_var, width=5).grid(row=0, column=1, sticky="w", padx=(5, 20))

        ttk.Label(hint_frame, text="Blur (px):").grid(row=0, column=2, sticky="w")
        self.blur_var = tk.IntVar(value=20)
        ttk.Spinbox(hint_frame, from_=0, to=200, textvariable=self.blur_var, width=5).grid(row=0, column=3, sticky="w", padx=(5, 0))

        self.hint_info = ttk.Label(hint_frame, text="Auto-generated from MP4 via chroma key + MatAnyone2", foreground="gray")
        self.hint_info.grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))

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

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.grid(row=5, column=0, columnspan=2, sticky="ew", **pad)

        self.log_text = tk.Text(log_frame, height=8, width=68, state="disabled", bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # --- Buttons ---
        btn_frame = ttk.Frame(self.root)
        btn_frame.grid(row=6, column=0, columnspan=2, pady=15)

        self.start_btn = ttk.Button(btn_frame, text="▶  Start", command=self._start)
        self.start_btn.grid(row=0, column=0, ipadx=20, ipady=5, padx=5)

        self.open_btn = ttk.Button(btn_frame, text="Open Output Folder", command=self._open_output, state="disabled")
        self.open_btn.grid(row=0, column=1, ipadx=10, ipady=5, padx=5)

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
        self.hint_info.config(text="Auto-generated from MP4 via chroma key + MatAnyone2", foreground="gray")
        self.info_label.config(text="✓  MP4 selected — AlphaHint will be auto-generated", foreground="green")

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder with Input frames (no auto-hint)")
        if not folder:
            return
        self.input_var.set(folder)
        frames = [f for f in os.listdir(folder) if is_image_file(f)]
        self.info_label.config(
            text=f"✓  {len(frames)} image frames detected (manual AlphaHint mode)",
            foreground="green" if frames else "orange"
        )
        self.hint_info.config(text="⚠  Folder mode: AlphaHint/ must exist alongside Input/ manually", foreground="orange")

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

        is_video = input_path.lower().endswith(VIDEO_EXTS)
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
                self._log(f"Input: {os.path.basename(input_path)}")
                self._log("Launching MatAnyone2...")

                cmd = [
                    MATANYONE2_PYTHON, GENERATE_HINTS_SCRIPT,
                    "-i", input_path,
                    "-o", alpha_dir,
                    "--export-input", input_dir,
                    "--dilate", str(self.dilate_var.get()),
                    "--blur", str(self.blur_var.get()),
                ]
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
                self._proc = proc

                def read_hints_stdout():
                    for line in proc.stdout:
                        self._log(line.rstrip())
                threading.Thread(target=read_hints_stdout, daemon=True).start()

                while proc.poll() is None:
                    done = len([f for f in os.listdir(alpha_dir) if is_image_file(f)]) if os.path.exists(alpha_dir) else 0
                    total = len([f for f in os.listdir(input_dir) if is_image_file(f)]) if os.path.exists(input_dir) else 1
                    pct = min(done / max(total, 1) * 50, 50)  # phase 1 = 0–50%
                    self._update_progress(pct, 100, f"Step 1/2 — AlphaHint: {done} frames...", "blue")
                    time.sleep(0.5)

                if proc.returncode != 0:
                    raise RuntimeError("AlphaHint generation failed. Check log.")

                self._log("AlphaHint generation done.")
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
                    # flat folder: copy all images as input, no alpha
                    for f in sorted(os.listdir(input_path)):
                        if is_image_file(f):
                            shutil.copy2(os.path.join(input_path, f), input_dir)

            # Phase 2: CorridorKey inference
            total_frames = len([f for f in os.listdir(input_dir) if is_image_file(f)])
            self._update_progress(50, 100, f"Step 2/2 — Running CorridorKey ({total_frames} frames)...", "blue")
            self._log(f"Starting CorridorKey inference on {total_frames} frames...")

            import shutil as _shutil
            uv_path = _shutil.which("uv") or "uv"
            gamma = "s" if self.gamma_var.get() == "sRGB" else "l"
            despill = str(self.despill_var.get())
            despeckle = "y" if self.despeckle_var.get() else "n"
            despeckle_size = str(self.despeckle_size_var.get())
            refiner = f"{self.refiner_var.get():.1f}"
            stdin_input = "\n".join([gamma, despill, despeckle, despeckle_size, refiner]) + "\n"

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(
                [uv_path, "run", "python", "-u", "clip_manager.py", "--action", "run_inference", "--shot", full_shot_name],
                cwd=BASE_DIR,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            self._proc = proc
            self._log(f"CorridorKey PID {proc.pid}")

            def write_stdin():
                try:
                    proc.stdin.write(stdin_input)
                    proc.stdin.flush()
                    proc.stdin.close()
                except Exception:
                    pass
            threading.Thread(target=write_stdin, daemon=True).start()

            def read_ck_stdout():
                for line in proc.stdout:
                    self._log(line.rstrip())
            threading.Thread(target=read_ck_stdout, daemon=True).start()

            while proc.poll() is None:
                done = len([f for f in os.listdir(comp_dir) if is_image_file(f)]) if os.path.exists(comp_dir) else 0
                pct = 50 + min(done / max(total_frames, 1) * 50, 50)  # phase 2 = 50–100%
                self._update_progress(pct, 100, f"Step 2/2 — CorridorKey: {done} / {total_frames} frames...", "blue")
                time.sleep(0.5)

            done = len([f for f in os.listdir(comp_dir) if is_image_file(f)]) if os.path.exists(comp_dir) else 0
            self._update_progress(100, 100, f"Done! {done} frames → {full_shot_name}", "green")
            self._log(f"✓ Finished! Output: {shot_path}")
            self._last_output_path = shot_path
            self.root.after(0, lambda: self.open_btn.config(state="normal"))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            self._update_progress(0, 100, "Failed — see log", "red")
        finally:
            self._proc = None
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

    def _open_output(self):
        if self._last_output_path and os.path.isdir(self._last_output_path):
            os.startfile(self._last_output_path)

    def _log(self, message):
        def append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, append)

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
