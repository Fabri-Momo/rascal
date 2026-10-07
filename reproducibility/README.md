# RASCAL — Reproducibility sessions

This directory contains the RASCAL session files (`.rasc`) used to reproduce the figures presented in the article.

The **`.rasc` session file** is the key element for reproducibility: it records the exact processing parameters (whitening matrix, rotation, push factor, etc.) applied to each image. Loading a `.rasc` file in RASCAL reproduces the result exactly, enabling independent verification of the published figures.

The corresponding images are located in the [`images_test/`](../images_test/) directory.

## Session files

| Session | Figure |
|---------|--------|
| `Fig_2.rasc` | Figure 2 |
| `Fig_3.rasc` | Figure 3 |
| `Fig_4.rasc` | Figure 4 |
| `Fig_5.rasc` | Figure 5 |
| `Fig_6b.rasc` | Figure 6b |
| `Fig_6f.rasc` | Figure 6f |
| `Fig_6i.rasc` | Figure 6i |
| `Fig_7c.rasc` | Figure 7c |
| `Fig_7d.rasc` | Figure 7d |
| `Fig_7e.rasc` | Figure 7e |
| `Fig_8b.rasc` | Figure 8b |
| `Fig_8c.rasc` | Figure 8c |
| `Fig_9.rasc` | Figure 9 |

## How to use

The `.rasc` session files are only compatible with the **Desktop version** of RASCAL.

1. Open RASCAL Desktop.
2. Use the **Open session** command and select the `.rasc` file.
3. The exact state used to generate the corresponding figure will be restored.

## Requirements

The Desktop version of RASCAL is required. Source code and installers are available in the [`desktop/`](../desktop/) directory, and pre-built packages in [`download/`](../download/).
