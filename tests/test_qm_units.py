from qmldd.qm.pyscf_runner import BOHR_TO_ANGSTROM, HARTREE_TO_EV


def test_conversion_constants_are_positive():
    assert HARTREE_TO_EV > 20
    assert 0.5 < BOHR_TO_ANGSTROM < 0.6
