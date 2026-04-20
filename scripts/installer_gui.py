#!/usr/bin/env python3
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
from pathlib import Path

class WasteXInstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WasteX Configuration Wizard")
        self.root.configure(padx=20, pady=20)
        # Removed fixed geometry so the window scales perfectly to its contents!

        self.base_dir = Path(__file__).resolve().parent.parent

        # Try to read existing .env
        env_path = self.base_dir / ".env"
        default_media = str(self.base_dir / "media")
        default_datasets = str(self.base_dir / "datasets")
        
        if env_path.exists():
            with open(env_path, "r") as f:
                for line in f:
                    if line.startswith("WASTE_MEDIA_ROOT="):
                        default_media = line.strip().split("=")[1]
                    elif line.startswith("WASTE_DATASETS_ROOT="):
                        default_datasets = line.strip().split("=")[1]

        self.media_path_var = tk.StringVar(value=default_media)
        self.datasets_path_var = tk.StringVar(value=default_datasets)

        self.create_widgets()

    def create_widgets(self):
        # Install Button (Pack this at the BOTTOM first, so it never gets pushed off screen)
        self.install_btn = tk.Button(self.root, text="Initialize System", bg="#10b981", fg="white", font=("Segoe UI", 12, "bold"), command=self.start_installation)
        self.install_btn.pack(side=tk.BOTTOM, fill=tk.X, pady=(15, 0))

        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="♻️ WasteX Initialization", font=("Segoe UI", 20, "bold")).pack(anchor=tk.W)
        ttk.Label(header_frame, text="Configure your storage directories.", font=("Segoe UI", 10)).pack(anchor=tk.W)
        
        # Info Box
        info_frame = tk.Frame(self.root, bg="#e0f2fe", padx=15, pady=10)
        info_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 20))
        tk.Label(info_frame, text="Guidance for Operators:\n• Edge Operators: Select your OOD Images Storage path.\n• Master Operators: Select your Datasets Storage path.", bg="#e0f2fe", fg="#0369a1", justify=tk.LEFT, font=("Segoe UI", 9)).pack(anchor=tk.W)

        # Paths Frame
        paths_frame = ttk.LabelFrame(self.root, text="Storage Configuration", padding=15)
        paths_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 20))

        # Media Path
        ttk.Label(paths_frame, text="📁 OOD Images Storage Path", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Entry(paths_frame, textvariable=self.media_path_var, width=50).grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Button(paths_frame, text="Browse", command=lambda: self.browse_folder(self.media_path_var)).grid(row=1, column=1)

        # Dataset Path
        ttk.Label(paths_frame, text="📊 Datasets Storage Path", font=("Segoe UI", 10, "bold")).grid(row=2, column=0, sticky=tk.W, pady=(15, 5))
        ttk.Entry(paths_frame, textvariable=self.datasets_path_var, width=50).grid(row=3, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Button(paths_frame, text="Browse", command=lambda: self.browse_folder(self.datasets_path_var)).grid(row=3, column=1)

        # Progress / Console area
        self.console_frame = ttk.LabelFrame(self.root, text="Installation Progress", padding=10)
        self.console_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.console_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.console_text = tk.Text(self.console_frame, height=8, bg="#1e293b", fg="#f8fafc", font=("Consolas", 9))
        self.console_text.pack(fill=tk.BOTH, expand=True)
        self.console_text.insert(tk.END, "Ready to initialize...\n")
        self.console_text.configure(state=tk.DISABLED)

    def browse_folder(self, var):
        folder = filedialog.askdirectory(initialdir=var.get(), title="Select Folder")
        if folder:
            var.set(folder)

    def log(self, message):
        self.console_text.configure(state=tk.NORMAL)
        self.console_text.insert(tk.END, message + "\n")
        self.console_text.see(tk.END)
        self.console_text.configure(state=tk.DISABLED)
        self.root.update()

    def set_progress(self, val):
        self.progress_var.set(val)
        self.root.update()

    def start_installation(self):
        if not messagebox.askyesno("Confirm", "Are you absolutely sure?\n\nThis will lock these paths into the environment."):
            return
            
        self.install_btn.configure(state=tk.DISABLED, bg="#94a3b8")
        threading.Thread(target=self.run_installation, daemon=True).start()

    def run_command_stream(self, cmd):
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(self.base_dir))
        for line in iter(process.stdout.readline, ''):
            self.log(line.strip())
        process.stdout.close()
        process.wait()
        return process.returncode

    def run_installation(self):
        try:
            self.set_progress(10)
            self.log("Initializing setup...")
            
            media_path = self.media_path_var.get().strip()
            datasets_path = self.datasets_path_var.get().strip()
            
            env_path = self.base_dir / ".env"
            
            # Read existing env
            env_content = ""
            if env_path.exists():
                with open(env_path, "r") as f:
                    env_content = f.read()
                    
            lines = [l for l in env_content.splitlines() if not l.startswith('WASTE_MEDIA_ROOT=') and not l.startswith('WASTE_DATASETS_ROOT=')]
            
            self.log("Creating directories...")
            if media_path:
                Path(media_path).mkdir(parents=True, exist_ok=True)
                lines.append(f"WASTE_MEDIA_ROOT={media_path}")
            if datasets_path:
                Path(datasets_path).mkdir(parents=True, exist_ok=True)
                lines.append(f"WASTE_DATASETS_ROOT={datasets_path}")
                
            self.set_progress(20)
            self.log("Writing configuration to .env...")
            with open(env_path, "w") as f:
                f.write('\n'.join(lines) + '\n')
                
            self.set_progress(40)
            self.log("Running Database Migrations... (This may take ~10 seconds while ML libraries load)")
            manage_py = str(self.base_dir / 'manage.py')
            
            ret = self.run_command_stream([sys.executable, manage_py, "migrate", "--noinput"])
            if ret != 0:
                raise Exception("Migrations failed!")
                
            self.set_progress(80)
            self.log("Setting up role groups...")
            ret = self.run_command_stream([sys.executable, manage_py, "setup_roles"])
            if ret != 0:
                raise Exception("Role setup failed!")
                
            self.set_progress(100)
            self.log("✅ Installation Complete!")
            self.root.after(0, lambda: messagebox.showinfo("Success", "System successfully initialized!\n\nYou can now close this window and run the Django server."))
            
        except Exception as e:
            self.log(f"❌ Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Installation failed:\n{str(e)}"))
        finally:
            self.root.after(0, lambda: self.install_btn.configure(state=tk.NORMAL, bg="#10b981", text="Finished"))


if __name__ == "__main__":
    # Ensure DPI awareness on Windows for crisp text
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass
        
    root = tk.Tk()
    
    # Modern theme
    style = ttk.Style()
    if 'vista' in style.theme_names():
        style.theme_use('vista')
    elif 'clam' in style.theme_names():
        style.theme_use('clam')
        
    app = WasteXInstallerGUI(root)
    root.mainloop()
