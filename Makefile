.PHONY: install test conformers xtb dft dataset train evaluate figures active parp

install:
	pip install -e ".[all]"

test:
	pytest -q

conformers:
	python scripts/00_make_conformers.py --config configs/quick.yaml

xtb:
	python scripts/01_run_xtb.py --config configs/quick.yaml

dft:
	python scripts/02_run_dft.py --config configs/quick.yaml --backend pyscf

dataset:
	python scripts/03_build_dataset.py --config configs/quick.yaml

train:
	python scripts/04_train.py --config configs/quick.yaml

evaluate:
	python scripts/05_evaluate.py --config configs/quick.yaml

figures:
	python scripts/08_make_figures.py --results results

active:
	python scripts/06_active_learn.py --config configs/quick.yaml

parp:
	python scripts/07_parp1_case_study.py
