# RASCAL — Installation Guide

RASCAL is developed and tested with Python 3.10. Python 3.10 or later is recommended.

Supported file formats for images are JPEG, PNG, BMP, TIFF (8 to 16-bit per channel, LZW-compressed or not), and RAW files from most camera manufacturers (Nikon, Canon, Sony, Fujifilm, Olympus, Panasonic, Pentax, etc.).

---

## Windows

Download and run the MSI installer. No additional steps required.

To run from source instead, install Python 3.10+, then:

```
python -m venv rascal_env
rascal_env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python Rascal.py
```

---

## Linux (from source)

### 1. System packages

Before installing the Python dependencies, make sure the required system packages for Python, virtual environments, Qt, and OpenGL are available. On Debian/Ubuntu-based systems:

```
sudo apt update
sudo apt install python3 python3-pip python3-venv libgl1 libglx-mesa0 libegl1 libxkbcommon-x11-0 libxcb-xinerama0
```

If Qt/OpenGL problems persist, the following additional packages may also be required:

```
sudo apt install libxcb-icccm4 libxcb-keysyms1 libxcb-render-util0 libxcb-cursor0
```

### 2. Python environment setup

```
python3 -m venv rascal_env
source rascal_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 Rascal.py
```

### 3. Subsequent uses

```
source rascal_env/bin/activate
python3 Rascal.py
```

### 4. Troubleshooting

- **imagecodecs** — On some systems (ARM, minimal distros), the pre-built wheel may not be available. Install the build dependencies first:
  ```
  sudo apt install liblzma-dev libjpeg-dev zlib1g-dev libtiff-dev build-essential
  pip install imagecodecs
  ```
  If it still fails, imagecodecs is not strictly required: RASCAL will work with uncompressed or standard TIFF files. Only LZW/Deflate-compressed TIFFs require this package.

- **rawpy** — On x86_64, the pip wheel includes LibRaw and should install without issues. On ARM or other architectures:
  ```
  sudo apt install libraw-dev
  pip install rawpy
  ```
  If rawpy cannot be installed, RASCAL will still work for all standard image formats (JPEG, PNG, BMP, TIFF). Only camera RAW files (NEF, CR2, ARW, etc.) require this package.

- **psutil** — Provides memory monitoring in the status bar. On most systems it installs without issues. If it fails:
  ```
  sudo apt install python3-dev build-essential
  pip install psutil
  ```
  If psutil is not available, RASCAL will run normally without the memory indicator.

- **PyQt5** — On some recent distributions (e.g. Ubuntu 24.04+), the pip PyQt5 wheel may conflict with system Qt libraries. In that case, use the system package instead:
  ```
  sudo apt install python3-pyqt5 python3-pyqt5.qtopengl
  ```
  and create your virtual environment with `--system-site-packages`:
  ```
  python3 -m venv --system-site-packages rascal_env
  ```

---

## macOS (from source)

### 1. Prerequisites

Install Python 3 via [python.org](https://www.python.org/downloads/macos/) or Homebrew:

```
brew install python@3.10
```

### 2. Python environment setup

```
python3 -m venv rascal_env
source rascal_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 Rascal.py
```

### 3. Subsequent uses

```
source rascal_env/bin/activate
python3 Rascal.py
```

### 4. Troubleshooting

- **imagecodecs** — Pre-built wheels are available for both Intel (x86_64) and Apple Silicon (arm64). If installation fails:
  ```
  brew install libtiff xz jpeg
  pip install imagecodecs
  ```
  If it still fails, RASCAL will work without it — only LZW/Deflate-compressed TIFFs require this package.

- **rawpy** — Wheels are available for macOS Intel and Apple Silicon since version 0.19. If installation fails:
  ```
  brew install libraw
  pip install rawpy
  ```
  Without rawpy, RASCAL will still open JPEG, PNG, BMP and TIFF files. Only camera RAW files (NEF, CR2, ARW, etc.) require this package.

- **psutil** — Usually installs without issues on macOS. If it fails, ensure Xcode command-line tools are available:
  ```
  xcode-select --install
  pip install psutil
  ```
  RASCAL will run normally without psutil; only the memory indicator in the status bar will be unavailable.

- **PyQt5 on Apple Silicon** — The official PyQt5 pip wheel does not support arm64 natively. Two solutions:
  1. Install via Homebrew:
     ```
     brew install pyqt@5
     python3 -m venv --system-site-packages rascal_env
     ```
  2. Or use Rosetta (x86_64 Python):
     ```
     arch -x86_64 /usr/local/bin/python3 -m venv rascal_env
     source rascal_env/bin/activate
     pip install -r requirements.txt
     ```

- **OpenGL warnings** — macOS has deprecated OpenGL since Catalina (10.15). RASCAL still works fine, but you may see deprecation warnings in the console. These are harmless and can be ignored.
