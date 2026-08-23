# Interview guide

## 1. Why delta learning?

A low-cost method can encode much of the qualitative physics while retaining systematic error. Learning only the discrepancy to a higher-fidelity method can reduce the complexity of the ML target and reduce the number of expensive labels needed. The important validation is whether the corrected model improves over the baseline on genuinely held-out molecules, not merely held-out conformers.

## 2. Why must splitting occur by molecule?

If conformers of the same molecule appear in both training and test sets, the model can exploit near-identical chemistry and geometry. A molecule-level split gives a more meaningful estimate of transfer to unseen molecules. A scaffold split is even more stringent for drug discovery.

## 3. Why predict forces from an energy gradient?

Forces are the negative coordinate gradient of the potential energy. Deriving them from a scalar learned energy imposes energy conservation and makes the predicted force field consistent with the energy surface.

## 4. Why equivariance?

A molecule's energy should not change when the entire molecule is translated or rotated. Vector and tensor features, however, should transform predictably. Equivariant networks encode these geometric symmetries in the architecture rather than expecting the model to discover them from augmented data.

## 5. Why xTB?

GFN-xTB methods are fast semi-empirical quantum methods suitable for generating approximate geometries, energies, forces, and electronic descriptors for many molecular conformers. They are useful as a baseline when higher-level calculations are too expensive for every candidate.

## 6. Why DFT, and what are its limitations?

DFT is much cheaper than high-level correlated wave-function methods and is widely used for molecular energetics. Results still depend on the exchange-correlation functional, basis set, dispersion treatment, numerical grids, molecular charge/spin state, and the property being targeted.

## 7. Why not call the PARP1 analysis a binding-affinity model?

The present pipeline models isolated-ligand energetics. Binding free energy additionally involves the protein, solvent, protonation and tautomer states, conformational reorganisation, enthalpy/entropy, and sampling. PARP inhibitor biological effects also include catalytic inhibition and context-dependent trapping mechanisms.

## 8. How would you test OOD behaviour?

- leave-one-scaffold-family-out evaluation,
- element-composition shift,
- size shift,
- high-strain conformers,
- uncertainty versus absolute error,
- distance in learned representation space,
- calibration of prediction intervals.

## 9. How would active learning reduce compute?

Run the cheap baseline broadly, train on an initial DFT subset, score unlabelled conformers with an ensemble, and spend the next DFT calculations on geometries with the highest epistemic uncertainty or expected information gain.

## 10. What would you improve for production?

- use a larger, versioned QM dataset,
- record full provenance of every calculation,
- use workflow orchestration and retry policies,
- containerise CPU/GPU environments,
- support distributed QM and model training,
- add strict physical and numerical QC,
- benchmark alternative equivariant architectures,
- establish reproducible OOD challenge sets,
- add protein-ligand and free-energy modules only after the ligand-energy layer is validated.
