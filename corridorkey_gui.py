"""
CorridorKey GUI
Simple tkinter frontend for running CorridorKey inference.
Usage: uv run python corridorkey_gui.py
"""

import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(BASE_DIR, "ClipsForInference")


def is_image_file(filename: str) -> bool:
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff"))


def detect_input_and_mask(folder: str):
    """Split files in folder into (input_files, mask_files) by 'mask' in filename."""
    all_files = sorted([f for f in os.listdir(folder) if is_image_file(f)])
    mask_files = [f for f in all_files if "mask" in f.lower()]
    input_files = [f for f in all_files if "mask" not in f.lower()]
    return input_files, mask_files


def prepare_shot_folder(source_folder: str, shot_name: str):
    """Create dated shot folder in ClipsForInference and move files."""
    date_str = datetime.now().strftime("%Y.%m.%d")
    full_shot_name = f"{date_str}_{shot_name}"
    shot_path = os.path.join(CLIPS_DIR, full_shot_name)
    input_dir = os.path.join(shot_path, "Input")
    alpha_dir = os.path.join(shot_path, "AlphaHint")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)

    input_files, mask_files = detect_input_and_mask(source_folder)

    for f in input_files:
        shutil.copy2(os.path.join(source_folder, f), os.path.join(input_dir, f))
    for f in mask_files:
        shutil.copy2(os.path.join(source_folder, f), os.path.join(alpha_dir, f))

    return shot_path, len(input_files), len(mask_files)


class CorridorKeyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CorridorKey")
        self.root.resizable(False, False)
        self._proc = None
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}

        # --- Folder Selection ---
        folder_frame = ttk.LabelFrame(self.root, text="Input Folder", padding=10)
        folder_frame.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)

        self.folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_var, width=50).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).grid(row=0, column=1)

        self.info_label = ttk.Label(folder_frame, text="", foreground="gray")
        self.info_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # --- Shot Name ---
        name_frame = ttk.LabelFrame(self.root, text="Shot Name", padding=10)
        name_frame.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)

        self.shot_name_var = tk.StringVar(value="shot")
        ttk.Entry(name_frame, textvariable=self.shot_name_var, width=40).grid(row=0, column=0, sticky="w")
        ttk.Label(name_frame, text="→ Will be saved as: YYYY.MM.DD_<name>", foreground="gray").grid(row=0, column=1, padx=10)

        # --- Settings ---
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding=10)
        settings_frame.grid(row=2, column=0, columnspan=2, sticky="ew", **pad)

        # Gamma
        ttk.Label(settings_frame, text="Gamma:").grid(row=0, column=0, sticky="w")
        self.gamma_var = tk.StringVar(value="sRGB")
        gamma_combo = ttk.Combobox(settings_frame, textvariable=self.gamma_var, values=["sRGB", "Linear"], state="readonly", width=10)
        gamma_combo.grid(row=0, column=1, sticky="w", padx=(5, 20))

        # Despill
        ttk.Label(settings_frame, text="Despill (0–10):").grid(row=0, column=2, sticky="w")
        self.despill_var = tk.IntVar(value=5)
        ttk.Spinbox(settings_frame, from_=0, to=10, textvariable=self.despill_var, width=5).grid(row=0, column=3, sticky="w", padx=(5, 20))

        # Despeckle
        ttk.Label(settings_frame, text="Auto-Despeckle:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.despeckle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.despeckle_var, command=self._toggle_despeckle).grid(row=1, column=1, sticky="w", pady=(8, 0))

        ttk.Label(settings_frame, text="Min Size (px):").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.despeckle_size_var = tk.IntVar(value=400)
        self.despeckle_size_spin = ttk.Spinbox(settings_frame, from_=0, to=9999, textvariable=self.despeckle_size_var, width=6)
        self.despeckle_size_spin.grid(row=1, column=3, sticky="w", padx=(5, 20), pady=(8, 0))

        # Refiner
        ttk.Label(settings_frame, text="Refiner Strength:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.refiner_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(settings_frame, from_=0.0, to=2.0, increment=0.1, textvariable=self.refiner_var, width=5, format="%.1f").grid(row=2, column=1, sticky="w", padx=(5, 0), pady=(8, 0))

        # --- Progress ---
        progress_frame = ttk.LabelFrame(self.root, text="Progress", padding=10)
        progress_frame.grid(row=3, column=0, columnspan=2, sticky="ew", **pad)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=480)
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        self.progress_label = ttk.Label(progress_frame, text="Ready", foreground="gray")
        self.progress_label.grid(row=1, column=0, sticky="w", pady=(5, 0))

        # --- Log ---
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.grid(row=4, column=0, columnspan=2, sticky="ew", **pad)

        self.log_text = tk.Text(log_frame, height=8, width=68, state="disabled", bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9))
        self.log_text.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # --- Start Button ---
        self.start_btn = ttk.Button(self.root, text="▶  Start Inference", command=self._start)
        self.start_btn.grid(row=5, column=0, columnspan=2, pady=15, ipadx=20, ipady=5)

    def _toggle_despeckle(self):
        state = "normal" if self.despeckle_var.get() else "disabled"
        self.despeckle_size_spin.config(state=state)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder with Input + Mask frames")
        if not folder:
            return
        self.folder_var.set(folder)
        input_files, mask_files = detect_input_and_mask(folder)
        if not input_files or not mask_files:
            self.info_label.config(
                text=f"⚠  Found {len(input_files)} input, {len(mask_files)} mask files — check filenames contain 'mask'",
                foreground="orange"
            )
        else:
            self.info_label.config(
                text=f"✓  {len(input_files)} input frames + {len(mask_files)} mask frames detected",
                foreground="green"
            )

    def _start(self):
        folder = self.folder_var.get().strip()
        shot_name = self.shot_name_var.get().strip() or "shot"

        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid input folder.")
            return

        input_files, mask_files = detect_input_and_mask(folder)
        if not input_files or not mask_files:
            messagebox.showerror("Error", f"Could not find input+mask pairs.\nInput: {len(input_files)}, Mask: {len(mask_files)}\nMake sure mask files have 'mask' in their filename.")
            return

        self.start_btn.config(state="disabled")
        self.progress_label.config(text="Preparing shot folder...", foreground="blue")
        self.root.update()

        threading.Thread(target=self._run, args=(folder, shot_name, len(input_files)), daemon=True).start()

    def _run(self, folder, shot_name, total_frames):
        try:
            # 1. Prepare folder structure
            shot_path, n_input, n_mask = prepare_shot_folder(folder, shot_name)
            comp_dir = os.path.join(shot_path, "Output", "Comp")

            self._update_progress(0, total_frames, "Loading model...")

            # 3. Launch inference subprocess
            uv_path = shutil.which("uv") or "uv"
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            gamma = "s" if self.gamma_var.get() == "sRGB" else "l"
            proc = subprocess.Popen(
                [
                    uv_path, "run", "python", "-u", "run_inference_direct.py",
                    "--gamma", gamma,
                    "--despill", str(self.despill_var.get()),
                    "--despeckle", "1" if self.despeckle_var.get() else "0",
                    "--despeckle_size", str(self.despeckle_size_var.get()),
                    "--refiner", f"{self.refiner_var.get():.1f}",
                    "--shot", os.path.basename(shot_path),
                ],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
            self._proc = proc
            self._log(f"Started PID {proc.pid} — loading model (~30s)...")

            # Read stdout in background thread to prevent pipe buffer deadlock
            import time

            def read_stdout():
                for line in proc.stdout:
                    self._log(line.rstrip())
            threading.Thread(target=read_stdout, daemon=True).start()

            # Monitor progress by polling output folder
            while proc.poll() is None:
                done = len([f for f in os.listdir(comp_dir) if is_image_file(f)]) if os.path.exists(comp_dir) else 0
                self._update_progress(done, total_frames, f"Frame {done} / {total_frames}...")
                time.sleep(0.5)

            done = len([f for f in os.listdir(comp_dir) if is_image_file(f)]) if os.path.exists(comp_dir) else 0
            self._update_progress(done, total_frames, f"Done! {done} frames")
            self._log(f"✓ Finished! {done} frames → {shot_path}")

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self._proc = None
            self.root.after(0, lambda: self.start_btn.config(state="normal"))

    def _log(self, message):
        def append():
            self.log_text.config(state="normal")
            self.log_text.insert("end", message + "\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        self.root.after(0, append)

    def _update_progress(self, done, total, message):
        pct = (done / total * 100) if total > 0 else 0
        color = "green" if done >= total and total > 0 else "blue"
        self.root.after(0, lambda: self.progress_var.set(pct))
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
