# RASCAL — Desktop version

Python implementation for Windows, macOS (Intel and Apple Silicon) and Linux.

## Contents

- **Python source code** — main application (`Rascal.py`) and modules
- **icons/** — application icons (`.ico`, `.png`, `.svg`)
- **shaders/** — GLSL shader files
- **packaging/** — build scripts (`Rascal.spec`, `build_msi.py`, `setup-mac.py`, `setup-linux.py`)
- **requirements.txt** — Python dependencies
- **INSTALL.md** — installation instructions for all platforms

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

Pre-built installers for all platforms are available in the [Releases](../releases/) directory.
