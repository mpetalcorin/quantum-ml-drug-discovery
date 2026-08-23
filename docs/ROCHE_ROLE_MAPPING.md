# Roche AI4DD role mapping

This file is intentionally recruiter-facing: each major capability in a quantum-chemistry/ML drug-discovery role is mapped to concrete repository evidence.

| Capability | Repository evidence |
|---|---|
| Quantum chemistry + ML | xTB baseline, DFT labels, Δ-learning objective |
| Psi4 / PySCF / ORCA / xTB | Dedicated backend adapters under `src/qmldd/qm/` |
| Python scientific programming | Package structure, command-line scripts, tests, configuration-driven workflows |
| PyTorch | Training loop, ensembles, autograd force corrections |
| E(3)-equivariant modelling | `E3NNDeltaEnergy` using e3nn spherical-harmonic irreps |
| Synthetic QM dataset generation | RDKit conformer generation followed by xTB and DFT calculations |
| Molecular energetics | Absolute and relative conformer energies |
| Forces | DFT/xTB force labels and learned Δ-force correction |
| Reproducibility | YAML configuration, deterministic seeds, molecule/scaffold splits, explicit backends |
| Scalability mindset | Cheap-baseline/high-fidelity-labelling separation and active-learning acquisition |
| Uncertainty | Deep ensemble standard deviation and error-correlation analysis |
| OOD evaluation | Production configuration uses scaffold-level split |
| Drug-discovery relevance | PARP1 inhibitor panel and explicit translational limitations |
| Scientific rigor | No claim that isolated-ligand QM energy equals protein binding or cellular response |
| Communication | README, interview guide, benchmark plots, scientific references |

## Strong portfolio claim

> Built a reproducible quantum-ML workflow that uses GFN2-xTB as a fast physical baseline, DFT as a higher-fidelity target, and an E(3)-equivariant graph network to learn energy and force corrections. Implemented molecule/scaffold holdouts, ensemble uncertainty, OOD evaluation, active learning, and a PARP1-focused drug-discovery case study.

## Claims to avoid until demonstrated with actual computed results

- Do not state that the model reaches DFT accuracy until the benchmark has been run.
- Do not state a speedup value until measured on the same hardware and molecular set.
- Do not state binding-affinity accuracy from the ligand-only pipeline.
- Do not state familiarity or fluency with a quantum-chemistry package solely because an adapter exists; run calculations, inspect failures, and be prepared to explain method choices.
