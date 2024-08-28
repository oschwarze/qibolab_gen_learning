import pytest

from qibolab import AcquisitionType, ExecutionParameters, create_platform
from qibolab.platform.platform import Platform
from qibolab.pulses import Delay, GaussianSquare, Pulse
from qibolab.sequence import PulseSequence

SWEPT_POINTS = 5


@pytest.fixture
def platform() -> Platform:
    return create_platform("dummy")


def test_dummy_initialization(platform: Platform):
    platform.connect()
    platform.disconnect()


def test_dummy_execute_coupler_pulse(platform: Platform):
    sequence = PulseSequence()

    channel = platform.get_coupler(0).flux
    pulse = Pulse(
        duration=30,
        amplitude=0.05,
        envelope=GaussianSquare(rel_sigma=5, width=0.75),
    )
    sequence.append((channel.name, pulse))

    options = ExecutionParameters(nshots=None)
    _ = platform.execute([sequence], options)


def test_dummy_execute_pulse_sequence_couplers():
    platform = create_platform("dummy")
    sequence = PulseSequence()

    natives = platform.natives
    cz = natives.two_qubit[(1, 2)].CZ.create_sequence()

    sequence.concatenate(cz)
    sequence.append((platform.qubits[0].probe.name, Delay(duration=40)))
    sequence.append((platform.qubits[2].probe.name, Delay(duration=40)))
    sequence.concatenate(natives.single_qubit[0].MZ.create_sequence())
    sequence.concatenate(natives.single_qubit[2].MZ.create_sequence())
    options = ExecutionParameters(nshots=None)
    _ = platform.execute([sequence], options)


@pytest.mark.parametrize(
    "acquisition", [AcquisitionType.INTEGRATION, AcquisitionType.DISCRIMINATION]
)
@pytest.mark.parametrize("batch_size", [None, 3, 5])
def test_dummy_execute_pulse_sequence_unrolling(
    platform: Platform, acquisition, batch_size
):
    nshots = 100
    nsequences = 10
    platform.instruments["dummy"].UNROLLING_BATCH_SIZE = batch_size
    natives = platform.natives
    sequences = []
    sequence = PulseSequence()
    sequence.concatenate(natives.single_qubit[0].MZ.create_sequence())
    for _ in range(nsequences):
        sequences.append(sequence)
    options = ExecutionParameters(nshots=nshots, acquisition_type=acquisition)
    result = platform.execute(sequences, options)
    assert len(next(iter(result.values()))) == nsequences
    for r in result[0]:
        if acquisition is AcquisitionType.INTEGRATION:
            assert r.magnitude.shape == (nshots,)
        if acquisition is AcquisitionType.DISCRIMINATION:
            assert r.samples.shape == (nshots,)
