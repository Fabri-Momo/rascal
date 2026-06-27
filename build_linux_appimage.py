# build_linux_appimage.py — Package the PyInstaller build as a portable AppImage
#
# Usage:
#   python3 build_linux_appimage.py
#
# Prerequisites:
#   Run setup_linux.py first to produce dist/Rascal/
#   Internet access (downloads appimagetool on first run)

import os
import stat
import shutil
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(ROOT, "dist", "Rascal")
APPDIR = os.path.join(ROOT, "dist", "Rascal.AppDir")
APPIMAGETOOL = os.path.join(ROOT, "appimagetool-x86_64.AppImage")
APPIMAGETOOL_URL = (
    "https://github.com/AppImage/AppImageKit/releases/download/continuous/"
    "appimagetool-x86_64.AppImage"
)
OUTPUT_APPIMAGE = os.path.join(ROOT, "dist", "Rascal-x86_64.AppImage")

# ── 1. Check PyInstaller output exists ───────────────────────────────────────
if not os.path.isfile(os.path.join(DIST_DIR, "Rascal")):
    raise SystemExit(
        "ERROR: dist/Rascal/Rascal not found.\n"
        "Run setup_linux.py first to build the PyInstaller bundle."
    )

# ── 2. Download appimagetool if needed ───────────────────────────────────────
if not os.path.isfile(APPIMAGETOOL):
    print(f"Downloading appimagetool from {APPIMAGETOOL_URL} ...")
    urllib.request.urlretrieve(APPIMAGETOOL_URL, APPIMAGETOOL)
    os.chmod(APPIMAGETOOL, os.stat(APPIMAGETOOL).st_mode | stat.S_IEXEC)
    print("Done.")
else:
    print("appimagetool already present, skipping download.")

# ── 3. Build AppDir structure ─────────────────────────────────────────────────
if os.path.exists(APPDIR):
    shutil.rmtree(APPDIR)
os.makedirs(APPDIR)

# Copy the entire PyInstaller bundle into AppDir/usr/bin/
usr_bin = os.path.join(APPDIR, "usr", "bin")
os.makedirs(usr_bin)
print(f"Copying {DIST_DIR} → {usr_bin} ...")
shutil.copytree(DIST_DIR, os.path.join(usr_bin, "Rascal"), dirs_exist_ok=True)

# Copy icon to AppDir root (required by AppImage spec)
icon_src = os.path.join(DIST_DIR, "_internal", "icon.svg")
icon_dst = os.path.join(APPDIR, "rascal.svg")
if os.path.isfile(icon_src):
    shutil.copy2(icon_src, icon_dst)
else:
    # Fallback: copy from project root
    shutil.copy2(os.path.join(ROOT, "icon.svg"), icon_dst)

# Write AppRun entry point
apprun_path = os.path.join(APPDIR, "AppRun")
with open(apprun_path, "w") as f:
    f.write(
        "#!/bin/bash\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'exec "$HERE/usr/bin/Rascal/Rascal" "$@"\n'
    )
os.chmod(apprun_path, 0o755)

# Write .desktop file at AppDir root (required by AppImage spec)
desktop_path = os.path.join(APPDIR, "rascal.desktop")
with open(desktop_path, "w") as f:
    f.write(
        "[Desktop Entry]\n"
        "Name=RASCAL\n"
        "Comment=ERA2 – Real-time Augmented Spectral Color Analysis\n"
        "Exec=Rascal\n"
        "Icon=rascal\n"
        "Terminal=false\n"
        "Type=Application\n"
        "Categories=Science;Graphics;\n"
    )

# ── 4. Run appimagetool ───────────────────────────────────────────────────────
os.makedirs(os.path.join(ROOT, "dist"), exist_ok=True)
print(f"Building AppImage → {OUTPUT_APPIMAGE} ...")
env = os.environ.copy()
env.setdefault("ARCH", "x86_64")
subprocess.run(
    [APPIMAGETOOL, APPDIR, OUTPUT_APPIMAGE],
    check=True,
    env=env,
)

os.chmod(OUTPUT_APPIMAGE, 0o755)
print(f"\nDone! Portable AppImage: {OUTPUT_APPIMAGE}")
print("Distribute this single file. Users just run:  ./Rascal-x86_64.AppImage")
