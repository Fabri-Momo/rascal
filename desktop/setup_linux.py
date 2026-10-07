# setup_linux.py — Build RASCAL as a standalone Linux executable using PyInstaller
#
# Usage:
#   python3 setup_linux.py
#
# Prerequisites:
#   pip install pyinstaller
#   (run inside the same virtual environment used for RASCAL)

import subprocess
import sys
import os

assets = [
    "rascal_splash.png",
    "rascal_clean.ico",
    "camera.svg",
    "file-upload.svg",
    "reset.svg",
    "icon.svg",
]

add_data = []
for a in assets:
    add_data += ["--add-data", f"{a}:."]

icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",
    "--windowed",
    "--name", "Rascal",
    "--collect-all", "vispy",
    "--collect-all", "imageio",
    "--collect-all", "rawpy",
    "--collect-all", "imagecodecs",
    "--collect-all", "psutil",
] + add_data + ["Rascal.py"]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# Create an install.sh script in the dist folder that resolves paths at install time
dist_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "Rascal")
install_script = r"""#!/bin/bash
# RASCAL install script — registers RASCAL in the application menu
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXEC="$SCRIPT_DIR/Rascal"
ICON="$SCRIPT_DIR/_internal/icon.svg"
DESKTOP_FILE="$HOME/.local/share/applications/rascal.desktop"

chmod +x "$EXEC"

mkdir -p "$HOME/.local/share/applications"
cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Name=RASCAL
Comment=ERA2 – Real-time Augmented Spectral Color Analysis
Exec=$EXEC
Icon=$ICON
Terminal=false
Type=Application
Categories=Science;Graphics;
EOF

echo "RASCAL installed. You can now launch it from your application menu."
echo "To run it directly: $EXEC"
"""

install_file = os.path.join(dist_dir, "install.sh")
try:
    with open(install_file, "w") as f:
        f.write(install_script)
    os.chmod(install_file, 0o755)
    print(f"Install script created: {install_file}")
    print("Distribute the 'dist/Rascal/' folder. Users run './install.sh' once to register the app.")
except Exception as e:
    print(f"Could not create install.sh: {e}")
    print(f"You can launch RASCAL directly: {os.path.join(dist_dir, 'Rascal')}")
