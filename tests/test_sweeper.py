import numpy as np
import pytest

from qibolab.platform import Qubit
from qibolab.pulses import Pulse, Rectangular
from qibolab.sweeper import ChannelParameter, Parameter, Sweeper


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_pulses(parameter):
    pulse = Pulse(
        duration=40,
        amplitude=0.1,
        frequency=1e9,
        envelope=Rectangular(),
        channel="channel",
    )
    if parameter is Parameter.amplitude:
        parameter_range = np.random.rand(10)
    else:
        parameter_range = np.random.randint(10, size=10)
    if parameter in ChannelParameter:
        with pytest.raises(ValueError):
            sweeper = Sweeper(parameter, parameter_range, [pulse])
    else:
        sweeper = Sweeper(parameter, parameter_range, [pulse])
        assert sweeper.parameter is parameter


@pytest.mark.parametrize("parameter", Parameter)
def test_sweeper_qubits(parameter):
    qubit = Qubit(0)
    parameter_range = np.random.randint(10, size=10)
    if parameter in ChannelParameter:
        sweeper = Sweeper(parameter, parameter_range, qubits=[qubit])
        assert sweeper.parameter is parameter
    else:
        with pytest.raises(ValueError):
            sweeper = Sweeper(parameter, parameter_range, qubits=[qubit])


def test_sweeper_errors():
    pulse = Pulse(
        duration=40,
        amplitude=0.1,
        frequency=1e9,
        envelope=Rectangular(),
        channel="channel",
    )
    qubit = Qubit(0)
    parameter_range = np.random.randint(10, size=10)
    with pytest.raises(ValueError):
        Sweeper(Parameter.frequency, parameter_range)
    with pytest.raises(ValueError):
        Sweeper(Parameter.frequency, parameter_range, [pulse], [qubit])
