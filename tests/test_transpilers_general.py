import networkx as nx
import numpy as np
import pytest
from qibo import gates
from qibo.models import Circuit

from qibolab.transpilers.gate_decompositions import TwoQubitNatives
from qibolab.transpilers.general_connectivity import Transpiler, can_execute


def generate_random_circuit(nqubits, ngates, seed=42):
    """Generate a random circuit with RX and CZ gates."""
    np.random.seed(seed)
    one_qubit_gates = [gates.RX, gates.RY, gates.RZ]
    two_qubit_gates = [gates.CZ, gates.CNOT, gates.SWAP]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta, trainable=False))
        else:
            circuit.add(gate(*q))
    return circuit


def special_connectivity(connectivity):
    """Return a TII harware chip connectivity as a networkx graph"""
    if connectivity == "21_qubits":
        Q = [f"q{i}" for i in range(21)]
        chip = nx.Graph()
        chip.add_nodes_from(Q)
        graph_list_h = [
            (Q[0], Q[1]),
            (Q[1], Q[2]),
            (Q[3], Q[4]),
            (Q[4], Q[5]),
            (Q[5], Q[6]),
            (Q[6], Q[7]),
            (Q[8], Q[9]),
            (Q[9], Q[10]),
            (Q[10], Q[11]),
            (Q[11], Q[12]),
            (Q[13], Q[14]),
            (Q[14], Q[15]),
            (Q[15], Q[16]),
            (Q[16], Q[17]),
            (Q[18], Q[19]),
            (Q[19], Q[20]),
        ]
        graph_list_v = [
            (Q[3], Q[8]),
            (Q[8], Q[13]),
            (Q[0], Q[4]),
            (Q[4], Q[9]),
            (Q[9], Q[14]),
            (Q[14], Q[18]),
            (Q[1], Q[5]),
            (Q[5], Q[10]),
            (Q[10], Q[15]),
            (Q[15], Q[19]),
            (Q[2], Q[6]),
            (Q[6], Q[11]),
            (Q[11], Q[16]),
            (Q[16], Q[20]),
            (Q[7], Q[12]),
            (Q[12], Q[17]),
        ]
        chip.add_edges_from(graph_list_h + graph_list_v)
        """
        Q=[f"A{i}" for i in range(1, 7)]
        Q.append([f"B{i}" for i in range(1, 6)])
        Q.append([f"C{i}" for i in range(1, 6)])
        Q.append([f"D{i}" for i in range(1, 6)])
        chip = nx.Graph()
        chip.add_nodes_from(Q)
        graph_list = [
            (Q[0], Q[1]),
            (Q[0], Q[2]),
            (Q[0], Q[20]),
            (Q[1], Q[3]),
            (Q[2], Q[4]),
            (Q[2], Q[19]),
            (Q[3], Q[4]),
            (Q[3], Q[8]),
            (Q[4], Q[6]),
            (Q[5], Q[2]),
            (Q[5], Q[8]),
            (Q[5], Q[18]),
            (Q[5], Q[13]),
            (Q[6], Q[8]),
            (Q[6], Q[7]),
            (Q[7], Q[9]),
            (Q[8], Q[9]),
            (Q[9], Q[10]),
            (Q[9], Q[13]),
            (Q[10], Q[11]),
            (Q[11], Q[13]),
            (Q[11], Q[12]),
            (Q[12], Q[14]),
            (Q[13], Q[14]),
            (Q[14], Q[18]),
            (Q[14], Q[15]),
            (Q[15], Q[16]),
            (Q[16], Q[18]),
            (Q[16], Q[17]),
            (Q[17], Q[19]),
            (Q[18], Q[19]),
            (Q[19], Q[20]),
        ]
        chip.add_edges_from(graph_list)
        """
    elif connectivity == "5_qubits":
        Q = [f"q{i}" for i in range(5)]
        chip = nx.Graph()
        chip.add_nodes_from(Q)
        graph_list = [
            (Q[0], Q[2]),
            (Q[1], Q[2]),
            (Q[3], Q[2]),
            (Q[4], Q[2]),
        ]
        chip.add_edges_from(graph_list)
    return chip


def test_can_execute():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Z(1))
    circuit.add(gates.CZ(2, 1))
    assert can_execute(circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits"))


# TODO not working, can execute must return false
def test_cannot_execute_connectivity():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    assert can_execute(circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits"))


def test_cannot_execute_native2q():
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 2))
    assert not can_execute(circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits"))


def test_cannot_execute_native1q():
    circuit = Circuit(5)
    circuit.add(gates.H(0))
    assert not can_execute(circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits"))


# TODO raise error, no return false
def test_cannot_execute_wrong_native():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 2))
    with pytest.raises(AttributeError):
        can_execute(circuit, two_qubit_natives=TwoQubitNatives.CNOT, connectivity=special_connectivity("5_qubits"))


def test_connectivity_and_samples():
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    assert transpiler.connectivity.number_of_nodes() == 21
    assert transpiler.init_samples == 20


def test_connectivity_setter_error():
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    with pytest.raises(TypeError):
        transpiler.connectivity = 1


def test_3q_error():
    circ = Circuit(3)
    circ.add(gates.TOFFOLI(0, 1, 2))
    transpiler = Transpiler(connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20)
    with pytest.raises(ValueError):
        transpiler.transpile(circ)


def test_insufficient_qubits():
    circ = generate_random_circuit(10, 10)
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="greedy", init_samples=20)
    with pytest.raises(ValueError):
        transpiler.transpile(circ)


@pytest.mark.parametrize("gates", [10, 50])
@pytest.mark.parametrize("qubits", [5, 21])
@pytest.mark.parametrize(
    "natives", [TwoQubitNatives.CZ, TwoQubitNatives.iSWAP, TwoQubitNatives.CZ | TwoQubitNatives.iSWAP]
)
def test_random_circuits(gates, qubits, natives):
    transpiler = Transpiler(
        connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20, two_qubit_natives=natives
    )
    circ = generate_random_circuit(nqubits=qubits, ngates=gates)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert len(initial_map) == 21 and len(final_map) == 21
    assert added_swaps >= 0
    assert can_execute(transpiled_circuit, two_qubit_natives=natives, connectivity=special_connectivity("5_qubits"))


def test_subgraph_init_simple():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="subgraph")
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(2, 1))
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert added_swaps == 0
    assert len(initial_map) == 5 and len(final_map) == 5
    assert can_execute(
        transpiled_circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits")
    )


def test_subgraph_init():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="subgraph")
    circ = generate_random_circuit(5, 50)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert added_swaps >= 0
    assert len(initial_map) == 5 and len(final_map) == 5
    assert can_execute(
        transpiled_circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits")
    )


def test_custom_mapping():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"))
    transpiler.custom_qubit_mapping([1, 2, 3, 4, 0])
    circ = generate_random_circuit(5, 20)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert added_swaps >= 0
    assert len(initial_map) == 5 and len(final_map) == 5
    assert can_execute(
        transpiled_circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits")
    )


def test_custom_connectivity():
    transpiler = Transpiler(connectivity=special_connectivity("5_qubits"), init_method="greedy", init_samples=20)
    circ = generate_random_circuit(5, 20)
    Q = [f"q{i}" for i in range(5)]
    chip = nx.Graph()
    chip.add_nodes_from(Q)
    graph_list = [
        (Q[0], Q[2]),
        (Q[1], Q[2]),
        (Q[3], Q[2]),
        (Q[4], Q[2]),
    ]
    chip.add_edges_from(graph_list)
    transpiler.connectivity = chip
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert added_swaps >= 0
    assert len(initial_map) == 5 and len(final_map) == 5
    assert can_execute(
        transpiled_circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits")
    )


def test_split_setter():
    with pytest.raises(ValueError):
        transpiler = Transpiler(
            connectivity=special_connectivity("5_qubits"), init_method="subgraph", sampling_split=2.0
        )


def test_split():
    transpiler = Transpiler(
        connectivity=special_connectivity("21_qubits"), init_method="greedy", init_samples=20, sampling_split=0.2
    )
    circ = generate_random_circuit(21, 50)
    transpiled_circuit, final_map, initial_map, added_swaps = transpiler.transpile(circ)
    assert added_swaps >= 0
    assert len(initial_map) == 21 and len(final_map) == 21
    assert can_execute(
        transpiled_circuit, two_qubit_natives=TwoQubitNatives.CZ, connectivity=special_connectivity("5_qubits")
    )
