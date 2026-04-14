# GUI Installation Architecture & Flow Diagrams

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WASTEX INSTALLATION SYSTEM              │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  LAYER 1: USER INTERACTION (GUI)                            │
│                                                              │
│  install_gui.bat                                            │
│  └─ Launches Python interpreter                            │
│     └─ Runs installer_gui.py                               │
│        ├─ Main window                                      │
│        ├─ Path input fields                               │
│        ├─ Browse buttons (file dialogs)                   │
│        ├─ Validation logic                                │
│        ├─ Confirmation dialogs                            │
│        ├─ Progress window                                 │
│        └─ Threading (background execution)                │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 2: INSTALLATION AUTOMATION (DJANGO)                  │
│                                                              │
│  Python subprocess calls:                                    │
│  ├─ pip install -r requirements.txt                        │
│  ├─ python manage.py migrate                               │
│  ├─ python manage.py initialize_paths                      │
│  └─ python manage.py runserver                             │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 3: APPLICATION SETUP                                 │
│                                                              │
│  ├─ Folder Creation                                        │
│  │  └─ media/ datasets/ models/                           │
│  ├─ Database Setup                                         │
│  │  └─ db.sqlite3 with all tables                         │
│  ├─ Dependency Installation                               │
│  │  └─ Django, TensorFlow, Keras, etc.                    │
│  ├─ Configuration Storage                                  │
│  │  └─ AppSettings table with paths                       │
│  └─ Server Startup                                         │
│     └─ Django development server                          │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  LAYER 4: APPLICATION READY                                 │
│                                                              │
│  ✅ WasteX running at http://127.0.0.1:8000/              │
│  ✅ Dashboard accessible                                    │
│  ✅ Ready to upload and classify images                    │
│  ✅ Configured paths saved in database                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 📊 Installation Process Flow

```
START
  │
  ├─→ [1] USER STARTS
  │       └─ Double-clicks install_gui.bat
  │       └─ Python launches installer_gui.py
  │
  ├─→ [2] MAIN WINDOW DISPLAYS
  │       └─ Title: "WasteX Installation"
  │       └─ Three input fields visible
  │       │   ├─ Media path
  │       │   ├─ Datasets path
  │       │   └─ Models path
  │       └─ Browse buttons
  │       └─ Install/Cancel buttons
  │
  ├─→ [3] USER CONFIGURES PATHS
  │       │
  │       ├─ Option A: Click Browse
  │       │  └─ File explorer opens
  │       │  └─ User selects folder
  │       │  └─ Path auto-fills
  │       │
  │       └─ Option B: Type path
  │          └─ User types folder path
  │          └─ Field updates
  │
  ├─→ [4] USER CLICKS INSTALL
  │       │
  │       └─→ VALIDATION CHECK
  │           │
  │           ├─ Is media path valid? 
  │           │  ├─ Yes ✓ → Continue
  │           │  └─ No ✗ → Show error, return to [3]
  │           │
  │           ├─ Is datasets path valid?
  │           │  ├─ Yes ✓ → Continue
  │           │  └─ No ✗ → Show error, return to [3]
  │           │
  │           ├─ Is models path valid?
  │           │  ├─ Yes ✓ → Continue
  │           │  └─ No ✗ → Show error, return to [3]
  │           │
  │           └─ Can we write to paths?
  │              ├─ Yes ✓ → Continue
  │              └─ No ✗ → Permission error, return to [3]
  │
  ├─→ [5] CONFIRMATION DIALOG
  │       └─ Show summary of paths
  │       └─ "These paths cannot be changed after"
  │       └─ Ask: "Are you sure?"
  │       │
  │       ├─ User clicks No
  │       │  └─ Return to [3]
  │       │
  │       └─ User clicks Yes
  │          └─ Continue to [6]
  │
  ├─→ [6] PROGRESS WINDOW OPENS
  │       │
  │       ├─→ Step 1: CREATE FOLDERS
  │       │   │ [██░░░░░░░░░░░░░░] 10%
  │       │   │ Creating folder structure...
  │       │   └─ ✓ Folders created
  │       │
  │       ├─→ Step 2: INSTALL DEPENDENCIES
  │       │   │ [████████░░░░░░░░░░] 40%
  │       │   │ Installing Python packages...
  │       │   │ (pip install -r requirements.txt)
  │       │   └─ ✓ Dependencies installed
  │       │
  │       ├─→ Step 3: SETUP DATABASE
  │       │   │ [████████████░░░░░░░] 65%
  │       │   │ Running database migrations...
  │       │   │ (python manage.py migrate)
  │       │   └─ ✓ Database ready
  │       │
  │       ├─→ Step 4: SAVE CONFIGURATION
  │       │   │ [██████████████░░░░░] 85%
  │       │   │ Saving path configuration...
  │       │   │ (python manage.py initialize_paths)
  │       │   └─ ✓ Configuration saved
  │       │
  │       └─→ Step 5: COMPLETE
  │           │ [████████████████████] 100%
  │           │ ✅ Installation Complete!
  │           └─ Status: "Ready to start server"
  │
  ├─→ [7] SUCCESS DIALOG
  │       └─ "Installation complete!"
  │       └─ "Start WasteX server now?"
  │       │
  │       ├─ User clicks No
  │       │  └─ GUI closes
  │       │  └─ Installation done
  │       │
  │       └─ User clicks Yes
  │          └─ Continue to [8]
  │
  ├─→ [8] SERVER STARTUP
  │       │
  │       ├─ Django server starts
  │       │  └─ Listening on http://127.0.0.1:8000/
  │       │
  │       ├─ Browser opens automatically
  │       │  └─ Navigates to http://127.0.0.1:8000/
  │       │
  │       └─ Dashboard loads
  │          └─ Welcome message
  │          └─ No images yet
  │          └─ Ready to use
  │
  └─→ [9] INSTALLATION COMPLETE ✅
         └─ WasteX running and accessible
         └─ User can start uploading images
         └─ All paths configured and saved
         └─ Database ready
         └─ Models loaded
```

---

## 🔄 GUI Component Interactions

```
┌─────────────────────────────────────────────────────────────┐
│           INSTALLER_GUI.PY COMPONENT DIAGRAM                │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────┐
│   Main Window (tk.Tk)            │
│  ┌────────────────────────────┐  │
│  │  Header Frame              │  │
│  │  ├─ Title Label            │  │
│  │  │  "WasteX Installation"  │  │
│  │  └─ Subtitle Label         │  │
│  │     "Configure storage..." │  │
│  └────────────────────────────┘  │
│  ┌────────────────────────────┐  │
│  │  Content Frame             │  │
│  │                            │  │
│  │  📁 Media Storage          │  │
│  │  ┌──────────┬──────────┐   │  │
│  │  │ Entry    │ Button   │   │  │
│  │  │ Field    │ Browse() │   │  │
│  │  └──────────┴──────────┘   │  │
│  │  │ Path Label               │  │
│  │  │                            │  │
│  │  │ 📊 Datasets Storage       │  │
│  │  │ ┌──────────┬──────────┐   │  │
│  │  │ │ Entry    │ Button   │   │  │
│  │  │ │ Field    │ Browse() │   │  │
│  │  │ └──────────┴──────────┘   │  │
│  │  │ Path Label               │  │
│  │  │                            │  │
│  │  │ 🤖 Models Storage         │  │
│  │  │ ┌──────────┬──────────┐   │  │
│  │  │ │ Entry    │ Button   │   │  │
│  │  │ │ Field    │ Browse() │   │  │
│  │  │ └──────────┴──────────┘   │  │
│  │  │ Path Label               │  │
│  │  └────────────────────────────┘  │
│  │  ┌────────────────────────────┐  │
│  │  │  Button Frame              │  │
│  │  │  ┌────────┐  ┌────────┐   │  │
│  │  │  │ Cancel │  │ Install│   │  │
│  │  │  └────────┘  └────────┘   │  │
│  │  └────────────────────────────┘  │
└──────────────────────────────────────┘
        │
        ├─→ User clicks Browse
        │   └─ Calls browse_folder()
        │      └─ Opens filedialog.askdirectory()
        │         └─ User selects folder
        │            └─ Path updates in field
        │
        ├─→ User clicks Install
        │   └─ Calls start_installation()
        │      ├─ validate_paths()
        │      │  ├─ Check paths exist
        │      │  ├─ Check write permission
        │      │  └─ Return True/False
        │      ├─ Show confirmation dialog
        │      └─ Call show_progress_window()
        │         └─ Create new Toplevel window
        │            ├─ Progress bar widget
        │            ├─ Log text widget
        │            └─ Status label
        │               └─ Run installation in thread
        │
        └─→ Installation Thread
            ├─ Create folders
            ├─ Install dependencies
            ├─ Run migrations
            ├─ Save configuration
            └─ Show success dialog
```

---

## 🧵 Threading Model

```
MAIN THREAD (GUI)                   INSTALLATION THREAD (Background)
═════════════════════════════════════════════════════════════════════

User clicks Install
  │
  ├─→ validate_paths()
  │   └─ Blocks briefly
  │
  ├─→ Show confirmation
  │   └─ Wait for user
  │
  ├─→ Show progress window ┐
  │   └─ Create widgets    │
  │                        │
  ├─→ Start thread ────────┼──→ [Start background thread]
  │   └─ install_thread         │
  │       = Thread(target=      │
  │         run_install,        │
  │         daemon=True)        │
  │       .start()              │
  │                             │ run_install() executes:
  ├─→ Return immediately        │ ├─ Create folders
  │   (GUI stays responsive)    │ │  └─ Update GUI: log.insert()
  │                             │ │  └─ Update: progress['value']=20
  │                             │ ├─ pip install
  │                             │ │  └─ Update GUI
  │                             │ │  └─ Update: progress['value']=40
  │                             │ ├─ manage.py migrate
  │   GUI listens for:          │ │  └─ Update GUI
  │   ├─ Button clicks          │ │  └─ Update: progress['value']=60
  │   ├─ Window close           │ ├─ initialize_paths
  │   └─ Text input             │ │  └─ Update GUI
  │                             │ │  └─ Update: progress['value']=100
  │   (responsive)              │ │
  │                             │ └─ Installation done
  │                             │    └─ Show success dialog
  │   Progress updates as ←─────┴─ GUI updates propagate
  │   thread progresses           back to main thread
  │
  └─→ Installation complete
      └─ User clicks "Yes" to start server
```

---

## 📈 Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│              USER INPUT → DATABASE FLOW                     │
└─────────────────────────────────────────────────────────────┘

USER INTERACTION
       │
       ├─→ Click Browse
       │   └─ filedialog.askdirectory()
       │      └─ Returns: "D:\MyData\media"
       │         └─ Updates: media_path variable
       │            └─ Displayed in Entry widget
       │
       ├─→ Click Install
       │   └─ Get values from StringVar:
       │      ├─ media_path.get() → "D:\MyData\media"
       │      ├─ datasets_path.get() → "C:\Users\...\datasets"
       │      └─ models_path.get() → "C:\...\models"
       │
       └─→ Validate
           └─ For each path:
              ├─ Path(path).mkdir(parents=True, exist_ok=True)
              │  └─ Creates folders if needed
              │
              ├─ test_file = path / ".wastex_test"
              │  └─ Creates test file
              │
              └─ test_file.unlink()
                 └─ Deletes test file
                    └─ If OK: ✓ Writable
                    └─ If Error: ✗ Permission denied

INSTALLATION
       │
       └─→ run_install() thread:
           ├─ subprocess.run("pip install -r requirements.txt")
           │  └─ Installs all Python packages
           │     └─ Updates: progress['value']
           │     └─ Updates: log.insert() with output
           │
           ├─ subprocess.run("python manage.py migrate")
           │  └─ Runs migrations
           │     └─ Creates database tables
           │     └─ Updates GUI progress
           │
           └─ subprocess.run("python manage.py initialize_paths ...")
              └─ Saves configuration
                 └─ AppSettings.set('media_root', path)
                 └─ AppSettings.set('datasets_root', path)
                 └─ AppSettings.set('models_root', path)
                    └─ Saved to database

DATABASE (db.sqlite3)
       │
       └─→ AppSettings table:
           ├─ id | key | value | created_at | updated_at
           ├─ 1 | media_root | D:\MyData\media | 2026-04-10 | 2026-04-10
           ├─ 2 | datasets_root | C:\Users\...\datasets | ... | ...
           └─ 3 | models_root | C:\...\models | ... | ...

RUNTIME (When app starts)
       │
       └─→ settings.py loads:
           ├─ get_storage_path('WASTE_MEDIA_ROOT', 'media_root', 'media')
           │  └─ Checks environment variable
           │  └─ Checks database AppSettings
           │  └─ Uses default if not found
           │     └─ Returns: D:\MyData\media
           │
           └─ MEDIA_ROOT = get_storage_path(...)
              └─ Django uses this path globally
                 └─ All image uploads go to D:\MyData\media
                 └─ All file operations use this path
```

---

## ✅ Validation Flowchart

```
User clicks Install
       │
       └─→ validate_paths()
           │
           ├─→ Is media_path empty?
           │   ├─ Yes → Show Error Dialog → Return False
           │   └─ No → Continue
           │
           ├─→ Is datasets_path empty?
           │   ├─ Yes → Show Error Dialog → Return False
           │   └─ No → Continue
           │
           ├─→ Is models_path empty?
           │   ├─ Yes → Show Error Dialog → Return False
           │   └─ No → Continue
           │
           ├─→ For each path:
           │   │
           │   ├─→ Try to create folder
           │   │   ├─ Path.mkdir(parents=True, exist_ok=True)
           │   │   │  ├─ Success → Continue
           │   │   │  └─ Error → Show Error Dialog → Return False
           │   │   │
           │   │   └─ Create test file
           │   │      ├─ Success → Continue
           │   │      └─ PermissionError → Show Error Dialog → Return False
           │   │
           │   └─→ Delete test file
           │
           └─→ All paths valid?
               ├─ Yes → Return True → Show Confirmation
               └─ No → Return False → Show Error
```

---

## 🎊 Summary

This architecture shows:

1. **User-Friendly Interface**: Clean GUI with browse buttons
2. **Layered Architecture**: UI → Automation → Application → Ready
3. **Validation**: Multiple checkpoints before proceeding
4. **Threading**: Responsive UI during long operations
5. **Real-Time Feedback**: Progress bar, log, status updates
6. **Data Flow**: User input → Validation → Database → Application Usage
7. **Error Handling**: Clear error messages at each step
8. **Professional UX**: Confirmation dialogs, success messages, auto-startup

**Result: Enterprise-quality installation experience!** ⭐⭐⭐⭐⭐
