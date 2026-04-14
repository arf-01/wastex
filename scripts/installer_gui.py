"""
WasteX GUI Installer
Professional installation wizard with graphical interface
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from tkinter import Tk, messagebox, filedialog
import tkinter as tk
from tkinter import ttk
import threading
import time


class WasteXInstallerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WasteX Installation Wizard")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Center window on screen
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
        
        self.base_dir = Path(__file__).parent
        self.media_path = tk.StringVar(value=str(Path.home() / "WasteX" / "media"))
        self.datasets_path = tk.StringVar(value=str(Path.home() / "WasteX" / "datasets"))
        self.models_path = tk.StringVar(value=str(Path.home() / "WasteX" / "models"))
        
        self.current_step = 0
        self.setup_ui()
        
    def setup_ui(self):
        """Create the installer UI"""
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = ttk.Label(header_frame, text="WasteX Installation", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(header_frame, text="Configure storage locations",
                                  font=('Arial', 10))
        subtitle_label.pack()
        
        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Media path
        ttk.Label(content_frame, text="📁 Uploaded Images Storage:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        media_frame = ttk.Frame(content_frame)
        media_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(media_frame, textvariable=self.media_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(media_frame, text="Browse", 
                  command=lambda: self.browse_folder(self.media_path)).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(content_frame, text="Location where uploaded images will be stored",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        
        # Datasets path
        ttk.Label(content_frame, text="📊 Training Datasets Storage:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))
        
        datasets_frame = ttk.Frame(content_frame)
        datasets_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(datasets_frame, textvariable=self.datasets_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(datasets_frame, text="Browse", 
                  command=lambda: self.browse_folder(self.datasets_path)).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(content_frame, text="Location where training datasets will be stored",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        
        # Models path
        ttk.Label(content_frame, text="🤖 ML Models Storage:", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(15, 5))
        
        models_frame = ttk.Frame(content_frame)
        models_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(models_frame, textvariable=self.models_path, width=50).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(models_frame, text="Browse", 
                  command=lambda: self.browse_folder(self.models_path)).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(content_frame, text="Location where ML models will be stored",
                 font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
        
        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Button(button_frame, text="Cancel", 
                  command=self.root.quit).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Install", 
                  command=self.start_installation).pack(side=tk.RIGHT, padx=5)
        
    def browse_folder(self, var):
        """Open folder browser dialog"""
        folder = filedialog.askdirectory(title=f"Select folder for {var}")
        if folder:
            var.set(folder)
    
    def validate_paths(self):
        """Validate all selected paths"""
        paths = {
            'Media': self.media_path.get(),
            'Datasets': self.datasets_path.get(),
            'Models': self.models_path.get()
        }
        
        for name, path in paths.items():
            if not path:
                messagebox.showerror("Error", f"{name} path cannot be empty")
                return False
            
            try:
                p = Path(path)
                # Try to create if doesn't exist
                p.mkdir(parents=True, exist_ok=True)
                
                # Check write permission
                test_file = p / ".wastex_test"
                test_file.touch()
                test_file.unlink()
                
            except PermissionError:
                messagebox.showerror("Error", f"No permission to write to {name} path:\n{path}")
                return False
            except Exception as e:
                messagebox.showerror("Error", f"Invalid {name} path:\n{str(e)}")
                return False
        
        return True
    
    def start_installation(self):
        """Start the installation process"""
        if not self.validate_paths():
            return
        
        # Show confirmation dialog
        summary = f"""Installation Summary:

📁 Images: {self.media_path.get()}
📊 Datasets: {self.datasets_path.get()}
🤖 Models: {self.models_path.get()}

These paths cannot be changed after installation.
Are you sure?"""
        
        if messagebox.askyesno("Confirm Installation", summary):
            self.show_progress_window()
    
    def show_progress_window(self):
        """Show installation progress window"""
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Installing WasteX")
        progress_window.geometry("500x250")
        progress_window.resizable(False, False)
        progress_window.grab_set()
        
        # Center on parent
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 250
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 125
        progress_window.geometry(f"+{x}+{y}")
        
        # Header
        ttk.Label(progress_window, text="Installing WasteX...", 
                 font=('Arial', 12, 'bold')).pack(pady=20)
        
        # Progress bar
        progress = ttk.Progressbar(progress_window, length=400, mode='determinate')
        progress.pack(pady=10, padx=20)
        
        # Status text
        status_var = tk.StringVar(value="Starting installation...")
        status_label = ttk.Label(progress_window, textvariable=status_var, 
                                font=('Arial', 9), foreground='blue')
        status_label.pack(pady=5)
        
        # Log text
        log_text = tk.Text(progress_window, height=8, width=60, font=('Courier', 8))
        log_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Disable close button during installation
        progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
        # Run installation in background thread
        def run_install():
            try:
                # Update folders
                log_text.insert(tk.END, "Creating folders...\n")
                log_text.see(tk.END)
                progress_window.update()
                
                Path(self.media_path.get()).mkdir(parents=True, exist_ok=True)
                Path(self.datasets_path.get()).mkdir(parents=True, exist_ok=True)
                Path(self.models_path.get()).mkdir(parents=True, exist_ok=True)
                
                progress['value'] = 20
                log_text.insert(tk.END, "✓ Folders created\n")
                status_var.set("Installing dependencies...")
                progress_window.update()
                
                # Install dependencies
                log_text.insert(tk.END, "Installing Python packages...\n")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "-r", 
                     str(self.base_dir / "requirements.txt")],
                    capture_output=True, text=True, cwd=str(self.base_dir)
                )
                
                if result.returncode != 0:
                    log_text.insert(tk.END, f"Warning: {result.stderr}\n")
                else:
                    log_text.insert(tk.END, "✓ Dependencies installed\n")
                
                progress['value'] = 40
                log_text.see(tk.END)
                progress_window.update()
                
                # Run migrations
                status_var.set("Setting up database...")
                log_text.insert(tk.END, "Running database migrations...\n")
                result = subprocess.run(
                    [sys.executable, "manage.py", "migrate", "--noinput"],
                    capture_output=True, text=True, cwd=str(self.base_dir)
                )
                
                if result.returncode == 0:
                    log_text.insert(tk.END, "✓ Database ready\n")
                else:
                    log_text.insert(tk.END, f"Error: {result.stderr}\n")
                
                progress['value'] = 60
                log_text.see(tk.END)
                progress_window.update()
                
                # Run initialize_paths command
                status_var.set("Saving configuration...")
                log_text.insert(tk.END, "Saving path configuration...\n")
                result = subprocess.run(
                    [sys.executable, "manage.py", "initialize_paths",
                     "--media-root", self.media_path.get(),
                     "--datasets-root", self.datasets_path.get(),
                     "--models-root", self.models_path.get()],
                    capture_output=True, text=True, cwd=str(self.base_dir)
                )
                
                if result.returncode == 0:
                    log_text.insert(tk.END, "✓ Configuration saved\n")
                else:
                    log_text.insert(tk.END, f"Error: {result.stderr}\n")
                
                progress['value'] = 100
                log_text.insert(tk.END, "\n✅ Installation Complete!\n")
                status_var.set("Installation successful!")
                log_text.see(tk.END)
                progress_window.update()
                
                # Ask to start server
                time.sleep(1)
                progress_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Re-enable close
                
                if messagebox.askyesno("Success", 
                    "Installation complete!\n\nStart WasteX server now?"):
                    log_text.insert(tk.END, "\nStarting server...\n")
                    log_text.see(tk.END)
                    progress_window.update()
                    
                    subprocess.Popen(
                        [sys.executable, "manage.py", "runserver"],
                        cwd=str(self.base_dir)
                    )
                    
                    messagebox.showinfo("Server Started", 
                        "WasteX server is running at:\n\nhttp://127.0.0.1:8000/")
                
                progress_window.after(1000, progress_window.destroy)
                
            except Exception as e:
                log_text.insert(tk.END, f"\n❌ Error: {str(e)}\n")
                log_text.see(tk.END)
                progress_window.update()
                messagebox.showerror("Installation Error", str(e))
                progress_window.protocol("WM_DELETE_WINDOW", progress_window.destroy)
        
        # Start installation thread
        install_thread = threading.Thread(target=run_install, daemon=True)
        install_thread.start()


def main():
    """Main entry point"""
    root = Tk()
    app = WasteXInstallerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
