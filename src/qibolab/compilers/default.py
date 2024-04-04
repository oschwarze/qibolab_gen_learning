"""Implementation of the default compiler.

Uses I, Z, RZ, U3, CZ, and M as the set of native gates.
"""

import math

from qibolab.pulses import PulseSequence


def identity_rule(gate, platform, qubit_map):
    """Identity gate skipped."""
    return PulseSequence(), {}


def z_rule(gate, platform, qubit_map):
    """Z gate applied virtually."""
    qubit = int(qubit_map[gate.target_qubits[0]])
    return PulseSequence(), {qubit: math.pi}


def rz_rule(gate, platform, qubit_map):
    """RZ gate applied virtually."""
    qubit = int(qubit_map[gate.target_qubits[0]])
    return PulseSequence(), {qubit: gate.parameters[0]}


def gpi2_rule(gate, platform, qubit_map):
    """Rule for GPI2."""
    qubit = int(qubit_map[gate.target_qubits[0]])
    theta = gate.parameters[0]
    sequence = PulseSequence()
    pulse = platform.create_RX90_pulse(qubit, start=0, relative_phase=theta)
    sequence.add(pulse)
    return sequence, {}


def gpi_rule(gate, platform, qubit_map):
    """Rule for GPI."""
    qubit = int(qubit_map[gate.target_qubits[0]])
    theta = gate.parameters[0]
    sequence = PulseSequence()
    # the following definition has a global phase difference compare to
    # to the matrix representation. See
    # https://github.com/qiboteam/qibolab/pull/804#pullrequestreview-1890205509
    # for more detail.
    pulse = platform.create_RX_pulse(qubit, start=0, relative_phase=theta)
    sequence.add(pulse)
    return sequence, {}


def u3_rule(gate, platform, qubit_map):
    """U3 applied as RZ-RX90-RZ-RX90-RZ."""
    qubit = int(qubit_map[gate.target_qubits[0]])
    # Transform gate to U3 and add pi/2-pulses
    theta, phi, lam = gate.parameters
    # apply RZ(lam)
    virtual_z_phases = {qubit: lam}
    sequence = PulseSequence()
    # Fetch pi/2 pulse from calibration
    RX90_pulse_1 = platform.create_RX90_pulse(
        qubit, start=0, relative_phase=virtual_z_phases[qubit]
    )
    # apply RX(pi/2)
    sequence.add(RX90_pulse_1)
    # apply RZ(theta)
    virtual_z_phases[qubit] += theta
    # Fetch pi/2 pulse from calibration
    RX90_pulse_2 = platform.create_RX90_pulse(
        qubit,
        start=RX90_pulse_1.finish,
        relative_phase=virtual_z_phases[qubit] - math.pi,
    )
    # apply RX(-pi/2)
    sequence.add(RX90_pulse_2)
    # apply RZ(phi)
    virtual_z_phases[qubit] += phi

    return sequence, virtual_z_phases


def cz_rule(gate, platform, qubit_map):
    """CZ applied as defined in the platform runcard.

    Applying the CZ gate may involve sending pulses on qubits that the
    gate is not directly acting on.
    """
    qubits = (int(qubit_map[qubit]) for qubit in gates.qubits)
    return platform.create_CZ_pulse_sequence(qubits)


def cnot_rule(gate, platform, qubit_map):
    """CNOT applied as defined in the platform runcard."""
    qubits = (int(qubit_map[qubit]) for qubit in gates.qubits)
    return platform.create_CNOT_pulse_sequence(qubits)


def measurement_rule(gate, platform, qubit_map):
    """Measurement gate applied using the platform readout pulse."""
    sequence = PulseSequence()
    for qubit in gate.target_qubits:
        MZ_pulse = platform.create_MZ_pulse(int(qubit_map[qubit]), start=0)
        sequence.add(MZ_pulse)
    return sequence, {}
