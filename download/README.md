# Downloads

Ready-to-use distributions of RASCAL are available on the **[GitLab Releases page](https://gitlab.huma-num.fr/fmonna/rascal/-/releases)**.

This directory contains distributions that are versioned directly in the repository (ImageJ/Fiji plugins and Android APK).

## Available versions

| Platform | File type | Location | How to install |
|----------|-----------|----------|----------------|
| **Windows** | `.msi` installer | [GitLab Releases](https://gitlab.huma-num.fr/fmonna/rascal/-/releases) | Double-click the `.msi` and follow the installation wizard. Requires Windows 10 or later. |
| **Linux** | AppImage | [GitLab Releases](https://gitlab.huma-num.fr/fmonna/rascal/-/releases) | Make the AppImage executable with `chmod +x` and run it directly. |
| **macOS (Intel & Apple Silicon)** | `.dmg` | [GitLab Releases](https://gitlab.huma-num.fr/fmonna/rascal/-/releases) | Mount the `.dmg` and drag RASCAL into the Applications folder. |
| **Fiji** | `.jar` plugin | [`imagej/Rascal_GL.jar`](./imagej/Rascal_GL.jar) | Copy `Rascal_GL.jar` into the `plugins/` folder of Fiji and restart the application. |
| **ImageJ** | `.jar` (standalone) | [`imagej/Rascal_GL-standalone.jar`](./imagej/Rascal_GL-standalone.jar) | Copy `Rascal_GL-standalone.jar` into the `plugins/` folder of ImageJ and restart the application. |
| **Android** | `.apk` | [`android/rascal_1_8_7.apk`](./android/rascal_1_8_7.apk) | Enable *Install from unknown sources*, then open the `.apk` on your device. |

## Which version should I choose?

- **Desktop (Windows/macOS/Linux)** : complete reference implementation for scientific work, publication figures and textured 3D models.
- **HTML/WebGL** : demonstration and quick exploration without installation ([see `web/`](../web/)).
- **ImageJ/Fiji** : integration into existing scientific imaging workflows.
- **Android** : field exploration on phones and tablets.

## Detailed comparison

A detailed comparison of all implementations is available in [`../documentation/RASCAL_version_comparison.pdf`](../documentation/RASCAL_version_comparison.pdf).

## Documentation

Each implementation has its own documentation folder:

| Implementation | Documentation |
|----------------|---------------|
| Desktop (Windows/macOS/Linux) | [`../documentation/desktop/`](../documentation/desktop/) |
| ImageJ/Fiji plugin | [`../documentation/imagej/`](../documentation/imagej/) |
| Android | [`../documentation/android/`](../documentation/android/) |
| Web / WebGL | [`../documentation/web/`](../documentation/web/) |
| Linux-specific notes | [`../documentation/linux/`](../documentation/linux/) |

General documentation and user manuals are kept in [`../documentation/`](../documentation/).
