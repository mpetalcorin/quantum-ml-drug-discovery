import numpy as np


def test_force_delta_identity():
    xtb = np.array([[1.0, -0.5, 0.2]])
    dft = np.array([[1.4, -0.1, 0.0]])
    delta = dft - xtb
    reconstructed = xtb + delta
    np.testing.assert_allclose(reconstructed, dft)
