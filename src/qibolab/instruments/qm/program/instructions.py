from typing import Optional

from qm import qua
from qm.qua import declare, for_
from qualang_tools.loops import from_array

from qibolab.components import Config
from qibolab.execution_parameters import AcquisitionType, ExecutionParameters
from qibolab.pulses import Delay, PulseSequence
from qibolab.sweeper import ParallelSweepers, Sweeper

from ..config import operation
from .acquisition import Acquisition, Acquisitions
from .arguments import ExecutionArguments, Parameters
from .sweepers import QUA_SWEEPERS


def _delay(pulse: Delay, element: str, parameters: Parameters):
    # TODO: How to play delays on multiple elements?
    if parameters.duration is None:
        duration = pulse.duration // 4 + 1
    else:
        duration = parameters.duration
    qua.wait(duration, element)


def _play(
    op: str,
    element: str,
    parameters: Parameters,
    acquisition: Optional[Acquisition] = None,
):
    if parameters.phase is not None:
        qua.frame_rotation_2pi(parameters.phase, element)
    if parameters.amplitude is not None:
        op = op * parameters.amplitude

    if acquisition is not None:
        acquisition.measure(op)
    else:
        qua.play(op, element, duration=parameters.duration)

    if parameters.phase is not None:
        qua.reset_frame(element)


def play(args: ExecutionArguments):
    """Part of QUA program that plays an arbitrary pulse sequence.

    Should be used inside a ``program()`` context.
    """
    qua.align()
    for element, pulses in args.sequence.items():
        for pulse in pulses:
            op = operation(pulse)
            params = args.parameters[op]
            if isinstance(pulse, Delay):
                _delay(pulse, element, params)
            else:
                acquisition = args.acquisitions.get((op, element))
                _play(op, element, params, acquisition)

    if args.relaxation_time > 0:
        qua.wait(args.relaxation_time // 4)


def _sweep_recursion(sweepers, configs, args):
    """Unrolls a list of qibolab sweepers to the corresponding QUA for loops
    using recursion."""
    if len(sweepers) > 0:
        sweeper = sweepers[0]
        parameter = sweeper.parameter
        if parameter in QUA_SWEEPERS:
            qua_sweeper = QUA_SWEEPERS[parameter](sweeper)
        else:
            raise NotImplementedError(f"Sweeper for {parameter} is not implemented.")

        variable = qua_sweeper.declare()
        with for_(*from_array(variable, qua_sweeper.values)):
            qua_sweeper(variable, configs, args)
            _sweep_recursion(sweepers[1:], configs, args)

    else:
        play(args)


def sweep(
    sweepers: list[Sweeper], configs: dict[str, Config], args: ExecutionArguments
):
    """Public sweep function that is called by the driver."""
    # for sweeper in sweepers:
    #    if sweeper.parameter is Parameter.duration:
    #        _update_baked_pulses(sweeper, instructions, config)
    _sweep_recursion(sweepers, configs, args)


def program(
    self,
    configs: dict[str, Config],
    sequence: PulseSequence,
    options: ExecutionParameters,
    acquisitions: Acquisitions,
    sweepers: list[ParallelSweepers],
):
    """QUA program implementing the required experiment."""
    with qua.program() as experiment:
        n = declare(int)
        # declare acquisition variables
        for acquisition in acquisitions.values():
            acquisition.declare()
        # execute pulses
        args = ExecutionArguments(sequence, acquisitions, options.relaxation_time)
        with for_(n, 0, n < options.nshots, n + 1):
            sweep(list(sweepers), configs, args)
        # download acquisitions
        has_iq = options.acquisition_type is AcquisitionType.INTEGRATION
        buffer_dims = options.results_shape(sweepers)[::-1][int(has_iq) :]
        with qua.stream_processing():
            for acquisition in acquisitions.values():
                acquisition.download(*buffer_dims)
    return experiment
