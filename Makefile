.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/lt05_value_segmentation.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/lt05_value_segmentation.ipynb --ExecutePreprocessor.timeout=1800

reproduce: test notebook
