# Repository validation status

## Validated in the build environment

- All Python source files and scripts compile successfully.
- Four lightweight unit tests pass.
- RDKit conformer generation was executed successfully on the seed molecular panel.
- RDKit conformer generation was executed successfully on the four PARP inhibitor structures in the case-study panel.
- Molecule-level leakage prevention and uncertainty acquisition logic are unit tested.
- The force-correction identity, `F_DFT = F_xTB + ΔF`, is unit tested.

## Not claimed as executed in the build environment

The build environment did not contain PySCF, xtb-python, ASE, PyTorch Geometric, or e3nn, so no DFT/xTB benchmark numbers or trained equivariant-model performance are fabricated in this repository snapshot. The environment files and backend adapters are provided for execution on a scientific workstation or compute environment.

## Before showing benchmark claims in an application

Run the full pipeline and retain:

1. software versions and hardware information,
2. failed/converged calculation counts,
3. xTB versus DFT baseline errors,
4. ΔML energy and force errors,
5. molecule-level and scaffold-OOD performance,
6. conformer-ranking recovery,
7. uncertainty calibration,
8. measured wall-clock speedup,
9. active-learning learning curves across acquisition rounds.
