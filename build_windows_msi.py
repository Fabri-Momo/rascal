# build_msi.py — Builds Rascal.exe with PyInstaller then packages it as an MSI with WiX v3.
#
# Prerequisites (run once):
#   pip install pyinstaller
#   Install WiX v3: https://github.com/wixtoolset/wix3/releases/download/wix3141rtm/wix314.exe
#
# Usage:
#   python build_msi.py

import os
import sys
import uuid
import subprocess
import pathlib
import xml.etree.ElementTree as ET

# ── Configuration ──────────────────────────────────────────────────────────────
APP_NAME        = "Rascal"
APP_VERSION     = "1.7.1"
APP_MANUFACTURER = "Fabrice Monna"
UPGRADE_CODE    = "{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"  # keep stable across versions
ENTRY_SCRIPT    = "Rascal.py"
DIST_DIR        = pathlib.Path("dist") / APP_NAME
MSI_OUT         = pathlib.Path("dist") / f"{APP_NAME}-{APP_VERSION}.msi"
ICO_PATH        = "rascal_clean.ico"

ASSETS = [
    "rascal_splash.png",
    "sphere_cube_flat.png",
    "rascal_clean.ico",
    "camera.svg",
    "file-upload.svg",
    "reset.svg",
    "window-close.svg",
]

HERE = pathlib.Path(__file__).parent.resolve()


# ── Step 1 : PyInstaller ───────────────────────────────────────────────────────
def run_pyinstaller():
    print("=" * 60)
    print("Step 1 — PyInstaller")
    print("=" * 60)
    python_dir = os.path.dirname(sys.executable)
    add_data = []
    for a in ASSETS:
        add_data += ["--add-data", f"{a};."]
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name", APP_NAME,
        "--icon", str(HERE / ICO_PATH),
        "--paths", python_dir,
        "--paths", os.path.join(python_dir, "Library", "bin"),
        "--collect-all", "vispy",
        "--collect-all", "imageio",
        "--collect-all", "rawpy",
        "--collect-all", "imagecodecs",
        "--hidden-import", "rawpy",
        "--hidden-import", "imagecodecs",
        "--hidden-import", "psutil",
        "--collect-all", "Pillow",
    ] + add_data + [ENTRY_SCRIPT]
    subprocess.run(cmd, check=True, cwd=str(HERE))


# ── Step 2 : Generate .wxs ────────────────────────────────────────────────────
def _make_id(path: pathlib.Path, prefix: str) -> str:
    """Unique WiX id using uuid5 hash of the full path — always valid and unique."""
    h = str(uuid.uuid5(uuid.NAMESPACE_URL, str(path))).replace("-", "")
    return f"{prefix}_{h}"


def generate_wxs(wxs_path: pathlib.Path):
    print("=" * 60)
    print("Step 2 — Generating WXS")
    print("=" * 60)

    dist = HERE / DIST_DIR

    # Collect all files recursively
    all_files = sorted(dist.rglob("*"))

    # Build directory tree structure
    # dir_id_map: absolute dir path -> WiX dir id
    dir_id_map = {dist: "INSTALLDIR"}
    for f in all_files:
        if f.is_dir():
            rel = f.relative_to(dist)
            dir_id_map[f] = _make_id(rel, "dir")

    # XML namespaces (WiX v3)
    ns = "http://schemas.microsoft.com/wix/2006/wi"
    ET.register_namespace("", ns)

    def el(tag, **attrs):
        e = ET.Element(f"{{{ns}}}{tag}")
        for k, v in attrs.items():
            e.set(k, v)
        return e

    def sub(parent, tag, **attrs):
        e = el(tag, **attrs)
        parent.append(e)
        return e

    wix = el("Wix")

    product = sub(wix, "Product",
                  Name=APP_NAME,
                  Version=APP_VERSION,
                  Manufacturer=APP_MANUFACTURER,
                  UpgradeCode=UPGRADE_CODE,
                  Id="*",
                  Language="0")

    sub(product, "Package",
        InstallerVersion="405",
        Compressed="yes",
        InstallScope="perMachine",
        Platform="x64")

    # Allow upgrades
    sub(product, "MajorUpgrade",
        DowngradeErrorMessage="A newer version is already installed.")

    # Media
    sub(product, "Media", Id="1", Cabinet="cab1.cab", EmbedCab="yes")

    # Icon + shortcuts
    sub(product, "Icon", Id="AppIcon", SourceFile=str(HERE / ICO_PATH))
    sub(product, "Property", Id="ARPPRODUCTICON", Value="AppIcon")

    # UI
    sub(product, "UIRef", Id="WixUI_Minimal")
    sub(product, "UIRef", Id="WixUI_ErrorProgressText")
    sub(product, "WixVariable", Id="WixUILicenseRtf", Value="License.rtf")

    # Custom bitmaps: dialog background (493x312) and banner (493x58)
    dialog_bmp = HERE / "dist" / "wix_dialog.bmp"
    banner_bmp = HERE / "dist" / "wix_banner.bmp"
    _make_wix_bitmaps(HERE / ICO_PATH, dialog_bmp, banner_bmp)
    if dialog_bmp.exists():
        sub(product, "WixVariable", Id="WixUIDialogBmp", Value=str(dialog_bmp))
    if banner_bmp.exists():
        sub(product, "WixVariable", Id="WixUIBannerBmp",  Value=str(banner_bmp))

    # Feature
    feature = sub(product, "Feature", Id="ProductFeature", Title=APP_NAME, Level="1")

    # ── Directory tree ──
    dir_root = sub(product, "Directory", Id="TARGETDIR", Name="SourceDir")
    pf_dir = sub(dir_root, "Directory", Id="ProgramFiles64Folder")
    install_dir = sub(pf_dir, "Directory", Id="INSTALLDIR", Name=APP_NAME)
    sub(dir_root, "Directory", Id="DesktopFolder", Name="Desktop")

    # Helper: get or create sub-directory element
    dir_elem_map = {dist: install_dir}

    def get_dir_elem(d: pathlib.Path):
        if d in dir_elem_map:
            return dir_elem_map[d]
        parent_elem = get_dir_elem(d.parent)
        wix_id = dir_id_map.get(d, _make_id(d.relative_to(dist), "dir"))
        elem = sub(parent_elem, "Directory", Id=wix_id, Name=d.name)
        dir_elem_map[d] = elem
        return elem

    # ── Components ──
    comp_refs = []
    exe_comp_id = None

    for f in all_files:
        if not f.is_file():
            continue
        rel = f.relative_to(dist)
        comp_id = _make_id(rel, "comp")
        file_id = _make_id(rel, "file")
        dir_elem = get_dir_elem(f.parent)

        comp = sub(dir_elem, "Component",
                   Id=comp_id,
                   Guid=str(uuid.uuid5(uuid.NAMESPACE_URL, str(rel))).upper())

        file_elem = sub(comp, "File",
                        Id=file_id,
                        Source=str(f),
                        KeyPath="yes")

        if f.name == f"{APP_NAME}.exe" and f.parent == dist:
            exe_comp_id = comp_id

        comp_refs.append(comp_id)

    # Desktop shortcut in its own component (required for non-advertised + perMachine)
    sc_comp = sub(install_dir, "Component",
                  Id="comp_DesktopShortcut",
                  Guid=str(uuid.uuid5(uuid.NAMESPACE_URL, "DesktopShortcut")).upper())
    sub(sc_comp, "Shortcut",
        Id="DesktopShortcut",
        Directory="DesktopFolder",
        Name=APP_NAME,
        Target=f"[INSTALLDIR]{APP_NAME}.exe",
        WorkingDirectory="INSTALLDIR",
        Icon="AppIcon",
        IconIndex="0",
        Advertise="no")
    sub(sc_comp, "RemoveFolder", Id="RemoveDesktopShortcut", Directory="DesktopFolder", On="uninstall")
    sub(sc_comp, "RegistryValue",
        Root="HKCU",
        Key=f"Software\\{APP_MANUFACTURER}\\{APP_NAME}",
        Name="installed",
        Type="integer",
        Value="1",
        KeyPath="yes")
    comp_refs.append("comp_DesktopShortcut")

    for cid in comp_refs:
        sub(feature, "ComponentRef", Id=cid)

    # Write
    _ensure_license_rtf(HERE)
    tree = ET.ElementTree(wix)
    ET.indent(tree, space="  ")
    tree.write(str(wxs_path), encoding="utf-8", xml_declaration=True)
    print(f"WXS written → {wxs_path}")


# ── WiX bitmap helpers ───────────────────────────────────────────────────────
def _make_wix_bitmaps(ico_path: pathlib.Path, dialog_bmp: pathlib.Path, banner_bmp: pathlib.Path):
    """Generate WiX dialog (493x312) and banner (493x58) BMPs from the app icon."""
    dialog_bmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        icon = Image.open(str(ico_path)).convert("RGBA")

        # Remove near-black background from icon: make pixels darker than
        # threshold fully transparent so only the white wolf shape remains.
        import numpy as np
        icon_arr = np.array(icon)  # RGBA
        r, g, b, a = icon_arr[...,0], icon_arr[...,1], icon_arr[...,2], icon_arr[...,3]
        luminance = 0.299*r.astype(float) + 0.587*g.astype(float) + 0.114*b.astype(float)
        dark_mask = luminance < 60
        icon_arr[~dark_mask, 3] = 0  # make bright/white pixels transparent, keep dark fox
        from PIL import Image as _Image
        icon_clean = _Image.fromarray(icon_arr, "RGBA")

        # Dialog background (493x312): plain light grey.
        # Icon (black bg removed) placed in bottom-left corner where WiX never
        # renders controls (buttons are bottom-right, text is upper-centre).
        dw, dh = 493, 312
        dialog = _Image.new("RGB", (dw, dh), (240, 240, 240))
        icon_d = icon_clean.resize((80, 80), _Image.LANCZOS)
        dialog.paste(icon_d, (16, dh - 80 - 16), icon_d)
        dialog.save(str(dialog_bmp), "BMP")

        # Banner (493x58): plain light background, small icon far right.
        bw, bh = 493, 58
        banner = _Image.new("RGB", (bw, bh), (240, 240, 240))
        icon_b = icon_clean.resize((40, 40), _Image.LANCZOS)
        banner.paste(icon_b, (bw - 49, 9), icon_b)
        banner.save(str(banner_bmp), "BMP")

        print(f"WiX bitmaps generated: {dialog_bmp}, {banner_bmp}")
    except ImportError:
        print("Pillow not installed — WiX bitmaps skipped (plain installer UI).")
        # Remove variables so WiX uses defaults
        dialog_bmp.unlink(missing_ok=True)
        banner_bmp.unlink(missing_ok=True)


# ── License RTF helper ───────────────────────────────────────────────────────
def _txt_to_rtf_body(text: str) -> str:
    """Escape plain text for RTF embedding."""
    out = []
    for ch in text:
        if ch == '\\':
            out.append('\\\\')
        elif ch == '{':
            out.append('\\{')
        elif ch == '}':
            out.append('\\}')
        elif ch == '\n':
            out.append('\\line\n')
        elif ord(ch) > 127:
            out.append(f'\\u{ord(ch)}?')
        else:
            out.append(ch)
    return ''.join(out)


def _ensure_license_rtf(here: pathlib.Path):
    rtf_path = here / "License.rtf"

    MAIN_LICENSE = f"""{APP_NAME} — GNU General Public License v3
{'=' * 60}

Copyright (C) 2007 Free Software Foundation, Inc.
https://fsf.org/

This repository is distributed under the terms of the GNU General
Public License version 3 (GPL-3.0).

For the complete legal text of the GPL-3.0, please refer to:
https://www.gnu.org/licenses/gpl-3.0.txt

In summary:
  - You may use, copy, modify and distribute this software.
  - Any derivative work must also be distributed under GPL-3.0.
  - Source code must be made available when distributing binaries.
  - There is NO WARRANTY, to the extent permitted by law.
"""

    THIRD_PARTY = """\

Third-Party Licenses
====================

This software bundles or depends on the following open-source packages:

------------------------------------------------------------
NumPy — BSD 3-Clause License
Copyright (c) 2005-2024, NumPy Developers.
https://numpy.org/
------------------------------------------------------------

------------------------------------------------------------
PyQt5 — GPL v3 / Commercial
Copyright (C) Riverbank Computing Limited.
https://riverbankcomputing.com/software/pyqt/
------------------------------------------------------------

------------------------------------------------------------
PyOpenGL — BSD License
Copyright (c) PyOpenGL contributors.
http://pyopengl.sourceforge.net/
------------------------------------------------------------

------------------------------------------------------------
VisPy — BSD 2-Clause License
Copyright (c) 2013-2024, VisPy developers.
https://vispy.org/
------------------------------------------------------------

------------------------------------------------------------
imageio — BSD 2-Clause License
Copyright (c) 2014-2024, imageio contributors.
https://imageio.github.io/
------------------------------------------------------------

------------------------------------------------------------
rawpy — MIT License
Copyright (c) 2014 Uli Koehler.
https://github.com/letmaik/rawpy
------------------------------------------------------------

------------------------------------------------------------
Pillow (PIL Fork) — HPND License
Copyright (c) 1997-2011 by Secret Labs AB.
Copyright (c) 1995-2011 by Fredrik Lundh.
Copyright (c) 2010-2024 by Jeffrey A. Clark and contributors.
https://python-pillow.org/
------------------------------------------------------------

------------------------------------------------------------
psutil — BSD 3-Clause License
Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola.
https://github.com/giampaolo/psutil
------------------------------------------------------------

------------------------------------------------------------
SciPy — BSD 3-Clause License
Copyright (c) 2001-2024, SciPy Developers.
https://scipy.org/
------------------------------------------------------------
"""

    full_text = MAIN_LICENSE + THIRD_PARTY

    header = (
        r"{\rtf1\ansi\ansicpg1252\deff0"
        r"{\fonttbl{\f0\fswiss\fcharset0 Arial;}}"
        r"{\colortbl;\red0\green0\blue0;}"
        r"\f0\fs18\cf1 "
    )
    body = _txt_to_rtf_body(full_text)
    rtf_content = header + body + r"}"

    if rtf_path.exists():
        print(f"License.rtf already present — skipping regeneration ({rtf_path})")
        return
    rtf_path.write_text(rtf_content, encoding="ascii", errors="replace")
    print(f"License.rtf written → {rtf_path}")


# ── Step 3 : WiX build (v3: candle + light) ──────────────────────────────────
def run_wix(wxs_path: pathlib.Path):
    print("=" * 60)
    print("Step 3 — WiX build (MSI)")
    print("=" * 60)
    out = HERE / MSI_OUT
    wixobj = wxs_path.with_suffix(".wixobj")

    wix_ui_dir = _find_wix_ui_dir()

    # candle: compile wxs → wixobj
    candle_cmd = ["candle", str(wxs_path), "-out", str(wixobj),
                  "-arch", "x64", "-ext", "WixUIExtension"]
    print("Running:", " ".join(candle_cmd))
    subprocess.run(candle_cmd, check=True, cwd=str(HERE))

    # light: link wixobj → msi
    light_cmd = ["light", str(wixobj), "-out", str(out),
                 "-ext", "WixUIExtension", "-cultures:en-us", "-sice:ICE60"]
    print("Running:", " ".join(light_cmd))
    subprocess.run(light_cmd, check=True, cwd=str(HERE))
    print(f"\nMSI ready → {out}")


def _find_wix_ui_dir() -> str:
    """Try to locate the WiX v3 bin directory and add it to PATH if needed."""
    import shutil
    if shutil.which("candle"):
        return str(pathlib.Path(shutil.which("candle")).parent)
    # Common WiX v3 install locations
    candidates = sorted(
        pathlib.Path("C:/Program Files (x86)").glob("WiX Toolset*/bin"),
        reverse=True
    )
    if candidates:
        wix_bin = str(candidates[0])
        os.environ["PATH"] = wix_bin + os.pathsep + os.environ["PATH"]
        return wix_bin
    return ""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Rascal MSI installer")
    parser.add_argument("--skip-pyinstaller", action="store_true",
                        help="Skip PyInstaller step (use existing dist/Rascal/)")
    parser.add_argument("--skip-msi", action="store_true",
                        help="Only run PyInstaller, stop before WiX")
    args = parser.parse_args()

    if not args.skip_pyinstaller:
        run_pyinstaller()

    wxs_path = HERE / "dist" / f"{APP_NAME}.wxs"
    generate_wxs(wxs_path)

    if not args.skip_msi:
        run_wix(wxs_path)
