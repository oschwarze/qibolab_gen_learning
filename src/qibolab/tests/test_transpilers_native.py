import numpy as np
import pytest
from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit

from qibolab.transpilers.native import NativeGates


def assert_matrices_allclose(gate, phase=1):
    backend = NumpyBackend()
    native_gates = NativeGates()
    target_matrix = gate.asmatrix(backend)
    circuit = Circuit(len(gate.qubits))
    circuit.add(native_gates.translate_gate(gate))
    native_matrix = circuit.unitary(backend)
    np.testing.assert_allclose(native_matrix, phase * target_matrix, atol=1e-12)


@pytest.mark.parametrize("gatename", ["H", "X", "Y"])
def test_pauli_to_native(gatename):
    backend = NumpyBackend()
    gate = getattr(gates, gatename)(0)
    assert_matrices_allclose(gate, phase=-1j)


@pytest.mark.parametrize("gatename", ["RX", "RY", "RZ"])
def test_rotations_to_native(gatename):
    gate = getattr(gates, gatename)(0, theta=0.1)
    assert_matrices_allclose(gate)


def test_u2_to_native():
    gate = gates.U2(0, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate)


def test_u3_to_native():
    gate = gates.U3(0, theta=0.2, phi=0.1, lam=0.3)
    assert_matrices_allclose(gate)


@pytest.mark.parametrize("gatename", ["CNOT", "CZ", "SWAP", "FSWAP"])
def test_two_qubit_to_native(gatename):
    gate = getattr(gates, gatename)(0, 1)
    assert_matrices_allclose(gate)


@pytest.mark.parametrize("gatename", ["CRX", "CRY", "CRZ"])
def test_controlled_rotations_to_native(gatename):
    gate = getattr(gates, gatename)(0, 1, theta=0.1)
    assert_matrices_allclose(gate)


def test_cu1_to_native():
    gate = gates.CU1(0, 1, theta=0.4)
    assert_matrices_allclose(gate)


@pytest.mark.skip
def test_cu2_to_native():
    gate = gates.CU2(0, 1, phi=0.1, lam=0.2)
    assert_matrices_allclose(gate)


@pytest.mark.skip
def test_cu3_to_native():
    gate = gates.CU3(0, 1, theta=0.3, phi=0.1, lam=0.2)
    assert_matrices_allclose(gate)  # , phase=np.exp(0.3j / 2))


def test_fSim_to_native():
    gate = gates.fSim(0, 1, theta=0.3, phi=0.1)
    assert_matrices_allclose(gate)


def test_GeneralizedfSim_to_native():
    from qibolab.tests.test_transpilers_decompositions import random_unitary

    unitary = random_unitary(1)
    gate = gates.GeneralizedfSim(0, 1, unitary, phi=0.1)
    assert_matrices_allclose(gate)


@pytest.mark.parametrize("gatename", ["RXX", "RYY", "RZZ"])
def test_rnn_to_native(gatename):
    gate = getattr(gates, gatename)(0, 1, theta=0.1)
    assert_matrices_allclose(gate)


@pytest.mark.parametrize("nqubits", [1, 2])
def test_unitary_to_native(nqubits):
    from qibolab.tests.test_transpilers_decompositions import random_unitary

    u = random_unitary(nqubits)
    # transform to SU(2^nqubits) form
    u = u / np.sqrt(np.linalg.det(u))
    gate = gates.Unitary(u, *range(nqubits))
    assert_matrices_allclose(gate)