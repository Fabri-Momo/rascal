# py2exe est incompatible avec numpy 2.x (RecursionError dans hook_numpy).
# Utiliser PyInstaller à la place :
#   pip install pyinstaller
#   python setup_windows.py   (ce script lance pyinstaller automatiquement)

import subprocess
import sys
import os
assets = [
    "rascal_splash.png",
    "rascal_clean.ico",
    "camera.svg",
    "file-upload.svg",
    "reset.svg",
    "window-close.svg",
]

add_data = []
for a in assets:
    add_data += ["--add-data", f"{a};."]

python_dir = os.path.dirname(sys.executable)

ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rascal_clean.ico")

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--onedir",
    "--windowed",
    "--name", "Rascal",
    "--icon", ico_path,
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
    "--exclude-module", "scipy",
    "--exclude-module", "sklearn",
    "--exclude-module", "skimage",
    "--exclude-module", "sphinx",
    "--exclude-module", "IPython",
] + add_data + ["Rascal.py"]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)

# Créer le raccourci bureau avec l'icône correcte
import winreg, ctypes
try:
    import win32com.client
    exe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "Rascal", "Rascal.exe")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_path = os.path.join(desktop, "Rascal.lnk")
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = exe_path
    shortcut.WorkingDirectory = os.path.dirname(exe_path)
    shortcut.IconLocation = f"{ico_path},0"
    shortcut.save()
    print(f"Raccourci créé : {shortcut_path}")
except ImportError:
    print("pywin32 non installé — raccourci non créé automatiquement.")
    print(f"Crée manuellement un raccourci vers : dist\\Rascal\\Rascal.exe")
