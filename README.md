# RASCAL — Desktop version

Python implementation for Windows, macOS (Intel and Apple Silicon) and Linux.

## Contents

- **Rascal.py** — main application (single-file Python desktop GUI)
- **requirements.txt** — Python dependencies
- **rascal_clean.ico** — window icon
- **rascal_splash.png** — welcome page image
- **camera.svg**, **file-upload.svg**, **reset.svg** — toolbar icons
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

## Pre-built installers

Pre-built installers for all platforms are available in the [`download/`](../download/) directory.
