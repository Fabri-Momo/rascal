# setup_macos.py — Build RASCAL as a standalone macOS .app using PyInstaller
#
# Usage:
#   python3 setup_macos.py
#
# Prerequisites:
#   pip install pyinstaller
#   (run inside the same virtual environment used for RASCAL)

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()
APP_NAME = "Rascal"
ENTRY_SCRIPT = "Rascal.py"

# Assets used by the desktop application.
# Keep this list in sync with setup_windows.py / setup_linux.py.
ASSETS = [
    "rascal_splash.png",
    "sphere_cube_flat.png",
    "rascal_clean.ico",
    "camera.svg",
    "file-upload.svg",
    "reset.svg",
    "window-close.svg",
    "icon.svg",
]

# PyInstaller uses ":" as the --add-data separator on macOS/Linux.
add_data = []
for asset in ASSETS:
    asset_path = HERE / asset
    if asset_path.exists():
        add_data += ["--add-data", f"{asset}:."]
    else:
        print(f"WARNING: asset not found and will not be bundled: {asset}")

# macOS wants .icns for a native app icon.
# If you do not yet have rascal.icns, the build still works without --icon.
ICON_CANDIDATES = [
    HERE / "rascal.icns",
    HERE / "icon.icns",
]

icon_args = []
for icon_path in ICON_CANDIDATES:
    if icon_path.exists():
        icon_args = ["--icon", str(icon_path)]
        break

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--windowed",
    "--name", APP_NAME,

    # VisPy/OpenGL and image plugins are often loaded dynamically.
    "--collect-all", "vispy",
    "--collect-all", "imageio",
    "--collect-all", "rawpy",
    "--collect-all", "imagecodecs",
    "--collect-all", "psutil",
    "--collect-all", "PIL",

    # Extra safety for modules frequently missed by analysis.
    "--hidden-import", "rawpy",
    "--hidden-import", "imagecodecs",
    "--hidden-import", "psutil",
    "--hidden-import", "OpenGL",
    "--hidden-import", "OpenGL.GL",
    "--hidden-import", "vispy.app.backends._pyqt5",

    # Avoid bundling large unused scientific stacks if they are not needed.
    "--exclude-module", "sphinx",
    "--exclude-module", "IPython",
] + icon_args + add_data + [ENTRY_SCRIPT]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True, cwd=str(HERE))

app_path = HERE / "dist" / f"{APP_NAME}.app"
if not app_path.exists():
    raise SystemExit(f"ERROR: expected app not found: {app_path}")

print(f"\nDone! macOS application bundle: {app_path}")
