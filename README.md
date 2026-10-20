# facet-bgsp

[![arXiv](https://img.shields.io/badge/arXiv-2610.25721-b31b1b.svg)](https://arxiv.org/abs/2610.25721)
[![DOI](https://zenodo.org/badge/1424882738.svg)](https://doi.org/10.5281/zenodo.23460513)

Reproducibility companion for the paper

> **Corrected Block Sparse Spectral Optimization: Exact Models, Finite
> Identification, and Local Dynamics**
> Omid Solari (sAIm Research), 2026.

This repository contains the driver scripts, YAML configs, LaTeX source,
and smoke tests needed to regenerate every figure and table in the
paper. The underlying numerical library is published separately as
[`facet-core`](https://pypi.org/project/facet-core/) on PyPI (the bare
`facet` name is held by an unrelated project; the Python import name is
unchanged — `import facet`).

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'          # drivers + tests
python -m pip install -e '.[llm,scrna]'    # add transformers/torch + anndata extras
python -m pytest -q                        # run the smoke suite
```

The library itself (`facet-core>=0.1.1`, imported as `facet`) pins the algorithms the paper proves
properties about (A-SUM / A-SQ objectives, Safeguarded Riemannian
Ascent, safe-screening, support-aware refit). Everything in this
repository is a thin driver layer that calls into it.

## Repository layout

```text
experiments/   # one driver per figure (Q1..Q4, A4, T2..T3, M1..M10)
configs/       # deterministic YAML for each driver
runs/          # populated by driver output (CSV, PNG, provenance JSON)
tests/         # smoke tests exercising each driver on tiny fixtures
```

Paper source does not live in this repository; it is maintained in a
separate private Overleaf project and published as the arXiv preprint
cited in `CITATION.cff`.

## Reproducing a figure

Every figure in the paper's results section is regenerated from a
single YAML config.

```bash
# Classical / Tier-1
python experiments/q1_sum_vs_sq.py            --config configs/q1_sum_vs_sq.yaml           --output-dir runs/q1    --hero
python experiments/q3_margin_identification.py --config configs/q3_margin_identification.yaml --output-dir runs/q3  --hero
python experiments/q4_local_rate.py           --config configs/q4_local_rate.yaml          --output-dir runs/q4    --hero

# Modern-AI / M-panels
python experiments/m2_monosemanticity.py      --config configs/m2_monosemanticity.yaml     --output-dir runs/m2    --hero
python experiments/m4_facet_vs_sae.py         --config configs/m4_facet_vs_sae.yaml        --output-dir runs/m4    --hero
python experiments/m7_safe_screening.py       --config configs/m7_safe_screening.yaml      --output-dir runs/m7    --hero

# Real-world (M8 finance, M9 scRNA, M10 agentic AI)
python experiments/m8_finance.py              --config configs/m8_finance.yaml             --output-dir runs/m8    --hero
python experiments/m9_scrna.py                --config configs/m9_scrna.yaml               --output-dir runs/m9    --hero --interpretability
python experiments/m9_zheng_facs.py           --config configs/m9_zheng_facs.yaml          --output-dir runs/m9_zheng_facs    --hero --interpretability
python experiments/m9_zheng68k.py             --config configs/m9_zheng68k.yaml            --output-dir runs/m9_zheng68k      --hero --interpretability
python experiments/m9_tabula_muris.py         --config configs/m9_tabula_muris.yaml        --output-dir runs/m9_tabula_muris  --hero --interpretability
python experiments/m10_agentic.py             --config configs/m10_agentic.yaml            --output-dir runs/m10   --hero
```

Each driver writes a sidecar `<name>.provenance.json` next to the CSV so
every PNG in the paper is traceable back to the exact seed, config, and
cell / HVG / token counts used.

## First-run data downloads

Some drivers fetch real public datasets on first run and cache them
locally under `runs/`:

| Driver           | Dataset                            | Size      | Host                              |
|------------------|------------------------------------|-----------|-----------------------------------|
| `m8_finance`     | Fama-French 48 industry portfolios | ~0.6 MB   | Ken French data library           |
| `m9_scrna`       | 10x Genomics PBMC 3k               | ~8 MB     | `cf.10xgenomics.com`              |
| `m9_zheng_facs`  | Zheng17 bead-purified PBMCs        | ~150 MB   | `cf.10xgenomics.com` (8 samples)  |
| `m9_zheng68k`    | 10x fresh 68k PBMC donor A         | ~124 MB   | `cf.10xgenomics.com`              |
| `m9_tabula_muris`| Tabula Muris FACS v2               | ~290 MB   | figshare 5829687                  |

All downloads use `urllib` with a `curl` fallback so they work on networks
with TLS interception. Cached matrices live under
`runs/<driver>/_cache_*.npz` and `runs/<driver>/raw/`; subsequent runs
hit the cache instead of the network.

## Where the paper lives

LaTeX source is **not** in this repository. The paper is maintained in a
private Overleaf project and published on arXiv (identifier in
`CITATION.cff` once posted). This repository exists so an arXiv reader
can `pip install -e .` and regenerate every figure from the deterministic
YAML configs. The regenerated PNGs land under `runs/<driver>/`; the
exact artefacts included in the published PDF are tagged at this
repository's release that matches the arXiv version.

## Citing

If you use this repository or the library it depends on, please cite:

- The paper: [`arXiv:2610.25721`](https://arxiv.org/abs/2610.25721).
- The library release `facet-core==0.1.1` on PyPI.
- The companion repository via its Zenodo DOI
  [`10.5281/zenodo.23460513`](https://doi.org/10.5281/zenodo.23460513)
  (concept DOI; resolves to the latest version). For v1.0.0 pinning use
  [`10.5281/zenodo.23460512`](https://doi.org/10.5281/zenodo.23460512).

## License

MIT, see `LICENSE`.
