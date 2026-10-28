# Paper (IEEE conference format, two columns)

| File | What |
|---|---|
| `main.tex` | The paper (IEEEtran conference class) |
| `main.pdf` | Compiled paper (7 pages) |
| `refs.bib` | Bibliography; every entry checked against arXiv or the publisher |
| `make_figures.py` | Builds every figure in `figures/` and every data table in `tables/` from `results/crossover_v1/` and the MAE checkpoints, so no result number is typed by hand |
| `figures/*.pdf` | Generated figures |
| `tables/*.tex` | Generated table bodies, pulled into `main.tex` with `\input` |

## Rebuild figures and tables
From `mae-crossover`:
```
.venv\Scripts\python.exe paper\make_figures.py
```
To also get PNG previews, set the environment variable `PREVIEW_DIR` to a folder first.

## Compile
- **Overleaf:** New Project -> Upload Project -> choose `overleaf_upload.zip`, which contains `main.tex`, `refs.bib`,
  `figures/` and `tables/`. The zip is made by the command below and is not stored in git. Compiler: pdfLaTeX.
- **Locally**, with any TeX distribution: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
  On this laptop TinyTeX is in `%APPDATA%\TinyTeX\bin\windows` (see `docs/DOWNLOADS.md`).

To make the Overleaf zip (from `mae-crossover`):
```
.venv\Scripts\python.exe -c "import zipfile,pathlib; p=pathlib.Path('paper'); z=zipfile.ZipFile(p/'overleaf_upload.zip','w'); [z.write(f, f.relative_to(p)) for f in [p/'main.tex', p/'refs.bib', *p.glob('figures/*.pdf'), *p.glob('tables/*.tex')]]; z.close()"
```
