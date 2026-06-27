# RASCAL — Desktop version

Python implementation for Windows, macOS (Intel and Apple Silicon) and Linux.

## Contents

- **Rascal.py** — main application (single-file Python desktop GUI)
- **requirements.txt** — Python dependencies
- **rascal_clean.ico** — window icon
- **rascal_splash.png** — welcome page image
- **camera.svg**, **file-upload.svg**, **reset.svg**, **window-close.svg** — toolbar icons
- **icon.svg** — application icon (Linux)
- **sphere_cube_flat.png** — 3D view texture asset
- **setup_windows.py** — PyInstaller build script for Windows
- **build_windows_msi.py** — WiX MSI packaging script for Windows
- **setup_linux.py** — PyInstaller build script for Linux
- **build_linux_appimage.py** — AppImage packaging script for Linux
- **INSTALL.md** — detailed installation instructions for all platforms

## Requirements

See `INSTALL.md` for platform-specific instructions. Main dependencies:

```
numpy, imageio, imagecodecs, rawpy, psutil, PyQt5, PyOpenGL, vispy, Pillow
```

## Running from source

```bash
python -m venv rascal_env
# Windows:   rascal_env\Scripts\activate
# macOS/Linux: source rascal_env/bin/activate
pip install -r requirements.txt
python Rascal.py
```

## Building the Windows installer

On Windows, with the virtual environment activated and **PyInstaller** installed:

```bash
pip install pyinstaller
python setup_windows.py
python build_windows_msi.py --skip-pyinstaller
```

This produces `dist/Rascal-{version}.msi`. You also need **WiX v3** installed and available on PATH for the MSI step.

## Building the Linux AppImage

On Linux, with the virtual environment activated and **PyInstaller** installed:

```bash
pip install pyinstaller
python3 setup_linux.py
python3 build_linux_appimage.py
```

This produces `dist/Rascal-x86_64.AppImage`. The script downloads `appimagetool` automatically on first run.

## Pre-built installers

Pre-built installers for all platforms are available in the [`download/`](../download/) directory.
