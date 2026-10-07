# RASCAL — WebGL / HTML version

Browser-based implementation of RASCAL using WebGL. It runs entirely client-side without any server or installation.

## Contents

- **index.html** — Single-file application containing HTML, CSS and JavaScript
- **rascal_splash.png** — Splash image displayed in the toolbar

## Requirements

- A modern web browser supporting WebGL 2.0 (Chrome, Firefox, Edge, Safari)
- No build step or server is required

## How to run

### Online version

RASCAL is available online at: **https://fabricemonna.com/rascal/**

No installation or download required.

### Open directly in a browser

Simply open `index.html` in your browser. Some browsers may block local file access for WebGL textures; in that case, serve the file through a local HTTP server.

### Serve with a local HTTP server

Using Python:

```bash
python -m http.server 8080
```

Then navigate to `http://localhost:8080`.

## Features

- ZCA whitening
- Interactive colour-space rotation
- Radial Push enhancement
- ROI processing
- Preset management
- Session export

## Documentation

The WebGL user manual is available in [`../documentation/web/`](../documentation/web/).
