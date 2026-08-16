# Workflow: release

1. `make lint test` clean; slow integration tests run once on the target device.
2. Reproducibility: fresh venv via setup.sh → `python demo.py` works.
3. README results section regenerated from runs/ (scripts/make_readme_tables.py).
4. STATUS.md reflects reality; TODO.md pruned; BUGS.md open items acknowledged in README limitations.
5. Tag `vX.Y.Z`; push.
