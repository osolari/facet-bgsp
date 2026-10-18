"""the BGSP paper experiments.

Each experiment is a standalone Python module runnable as
``python -m papers.a.experiments.<name> --config configs/<name>.yaml``.
No shared framework; configs are plain YAML and outputs are CSV + an optional
PNG. The library code the experiment composes against lives in
``facet.papers.block_sparse`` and shared primitives under ``facet.``.
"""
