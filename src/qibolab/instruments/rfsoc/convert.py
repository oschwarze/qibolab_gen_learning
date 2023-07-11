"""Convert helper functions for rfsoc driver."""
from dataclasses import asdict
from typing import overload

import numpy as np
import qibosoq.components.base as rfsoc
import qibosoq.components.pulses as rfsoc_pulses

from qibolab.platform import Qubit
from qibolab.pulses import Pulse, PulseSequence, PulseShape
from qibolab.sweeper import BIAS, DURATION, START, Parameter, Sweeper

HZ_TO_MHZ = 1e-6
NS_TO_US = 1e-3


def replace_pulse_shape(rfsoc_pulse: rfsoc_pulses.Pulse, shape: PulseShape) -> rfsoc_pulses.Pulse:
    """Set pulse shape parameters in rfsoc_pulses pulse object."""
    if shape.name not in {"Gaussian", "Drag", "Rectangular"}:
        new_pulse = rfsoc_pulses.Arbitrary(
            **asdict(rfsoc_pulse), i_values=shape.envelope_waveform_i, q_values=shape.envelope_waveform_q
        )
        return new_pulse
    new_pulse = getattr(rfsoc_pulses, shape.name)(**asdict(rfsoc_pulse))
    if shape.name in {"Gaussian", "Drag"}:
        new_pulse.rel_sigma = shape.rel_sigma
        if shape.name == "Drag":
            new_pulse.beta = shape.beta
    return new_pulse


def pulse_lo_frequency(pulse: Pulse, qubits: dict[int, Qubit]) -> int:
    """Return local_oscillator frequency (HZ) of a pulse."""
    pulse_type = pulse.type.name.lower()
    try:
        lo_frequency = getattr(qubits[pulse.qubit], pulse_type).local_oscillator._frequency
    except AttributeError:
        lo_frequency = 0
    return lo_frequency


def convert_units_sweeper(sweeper: rfsoc.Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]):
    """Convert units for `qibosoq.abstract.Sweeper` considering also LOs."""
    for idx, jdx in enumerate(sweeper.indexes):
        parameter = sweeper.parameters[idx]
        if parameter is rfsoc.Parameter.FREQUENCY:
            pulse = sequence[jdx]
            lo_frequency = pulse_lo_frequency(pulse, qubits)

            sweeper.starts[idx] = (sweeper.starts[idx] - lo_frequency) * HZ_TO_MHZ
            sweeper.stops[idx] = (sweeper.stops[idx] - lo_frequency) * HZ_TO_MHZ
        elif parameter is rfsoc.Parameter.DELAY:
            sweeper.starts[idx] = sweeper.starts[idx] * NS_TO_US
            sweeper.stops[idx] = sweeper.stops[idx] * NS_TO_US
        elif parameter is rfsoc.Parameter.RELATIVE_PHASE:
            sweeper.starts[idx] = np.degrees(sweeper.starts[idx])
            sweeper.stops[idx] = np.degrees(sweeper.stops[idx])


def convert_qubit(qubit: Qubit) -> rfsoc.Qubit:
    """Convert `qibolab.platforms.abstract.Qubit` to `qibosoq.abstract.Qubit`."""
    if qubit.flux:
        return rfsoc.Qubit(qubit.flux.offset, qubit.flux.port.name)
    return rfsoc.Qubit(0.0, None)


def convert_pulse_sequence(sequence: PulseSequence, qubits: dict[int, Qubit]) -> list[rfsoc_pulses.Pulse]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""

    abs_time = 0
    list_sequence = []
    for pulse in sequence:
        abs_start = pulse.start * NS_TO_US
        start_delay = abs_start - abs_time
        pulse_dict = asdict(convert_pulse(pulse, qubits, start_delay))
        list_sequence.append(pulse_dict)

        abs_time += start_delay
    return list_sequence


def convert_pulse(pulse: Pulse, qubits: dict[int, Qubit], start_delay: float) -> rfsoc_pulses.Pulse:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""
    pulse_type = pulse.type.name.lower()
    dac = getattr(qubits[pulse.qubit], pulse_type).port.name
    adc = qubits[pulse.qubit].feedback.port.name if pulse_type == "readout" else None
    lo_frequency = pulse_lo_frequency(pulse, qubits)

    rfsoc_pulse = rfsoc_pulses.Pulse(
        frequency=(pulse.frequency - lo_frequency) * HZ_TO_MHZ,
        amplitude=pulse.amplitude,
        relative_phase=np.degrees(pulse.relative_phase),
        start_delay=start_delay,
        duration=pulse.duration * NS_TO_US,
        dac=dac,
        adc=adc,
        name=pulse.serial,
        type=pulse_type,
    )
    return replace_pulse_shape(rfsoc_pulse, pulse.shape)


def convert_parameter(par: Parameter) -> rfsoc.Parameter:
    """Convert a qibolab sweeper.Parameter into a qibosoq.Parameter."""
    return getattr(rfsoc.Parameter, par.name.upper())


def convert_sweep(sweeper: Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]) -> rfsoc.Sweeper:
    """Convert `qibolab.sweeper.Sweeper` to `qibosoq.abstract.Sweeper`.

    Note that any unit conversion is not done in this function (to avoid to do it multiple times).
    Conversion will be done in `convert_units_sweeper`.
    """
    parameters = []
    starts = []
    stops = []
    indexes = []

    if sweeper.parameter is BIAS:
        for qubit in sweeper.qubits:
            parameters.append(rfsoc.Parameter.BIAS)
            indexes.append(list(qubits.values()).index(qubit))

            base_value = qubit.flux.offset
            values = sweeper.get_values(base_value)
            starts.append(values[0])
            stops.append(values[-1])

        if max(np.abs(starts)) > 1 or max(np.abs(stops)) > 1:
            raise ValueError("Sweeper amplitude is set to reach values higher than 1")
    else:
        for pulse in sweeper.pulses:
            idx_sweep = sequence.index(pulse)
            indexes.append(idx_sweep)
            base_value = getattr(pulse, sweeper.parameter.name)
            if idx_sweep != 0 and sweeper.parameter is START:
                # do the conversion from start to delay
                base_value = base_value - sequence[idx_sweep - 1].start
            values = sweeper.get_values(base_value)
            starts.append(values[0])
            stops.append(values[-1])

            if sweeper.parameter is START:
                parameters.append(rfsoc.Parameter.DELAY)
            elif sweeper.parameter is DURATION:
                parameters.append(rfsoc.Parameter.DURATION)
                delta_start = values[0] - base_value
                delta_stop = values[-1] - base_value

                if len(sequence) > idx_sweep + 1:
                    # if duration-swept pulse is not last
                    indexes.append(idx_sweep + 1)
                    t_start = sequence[idx_sweep + 1].start - sequence[idx_sweep].start
                    parameters.append(rfsoc.Parameter.DELAY)
                    starts.append(t_start + delta_start)
                    stops.append(t_start + delta_stop)
            else:
                parameters.append(convert_parameter(sweeper.parameter))

    return rfsoc.Sweeper(
        parameters=parameters,
        indexes=indexes,
        starts=starts,
        stops=stops,
        expts=len(sweeper.values),
    )


@overload
def convert(qubit: Qubit) -> rfsoc.Qubit:
    """Convert `qibolab.platforms.abstract.Qubit` to `qibosoq.abstract.Qubit`."""


@overload
def convert(sequence: PulseSequence, qubits: dict[int, Qubit]) -> list[rfsoc_pulses.Pulse]:
    """Convert PulseSequence to list of rfosc pulses with relative time."""


@overload
def convert(pulse: Pulse, qubits: dict[int, Qubit], start_delay: float) -> rfsoc_pulses.Pulse:
    """Convert `qibolab.pulses.pulse` to `qibosoq.abstract.Pulse`."""


@overload
def convert(par: Parameter) -> rfsoc.Parameter:
    """Convert a qibolab sweeper.Parameter into a qibosoq.Parameter."""


@overload
def convert(sweeper: Sweeper, sequence: PulseSequence, qubits: dict[int, Qubit]) -> rfsoc.Sweeper:
    """Convert `qibolab.sweeper.Sweeper` to `qibosoq.abstract.Sweeper`."""


def convert(*args):
    """Convert from qibolab obj to qibosoq obj, overloaded."""
    if isinstance(args[0], Qubit):
        return convert_qubit(*args)
    if isinstance(args[0], Parameter):
        return convert_parameter(*args)
    if isinstance(args[0], PulseSequence):
        return convert_pulse_sequence(*args)
    if isinstance(args[0], Pulse):
        return convert_pulse(*args)
    if isinstance(args[0], Sweeper):
        return convert_sweep(*args)
    raise ValueError(f"Convert function received bad parameters ({type(args[0])}).")