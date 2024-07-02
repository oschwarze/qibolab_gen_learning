import json
<<<<<<< HEAD
import operator
from functools import reduce
from pathlib import Path

=======
from pathlib import Path

GHz = 1e9
ns = 1e-9

>>>>>>> da0bddaf5ad2c76501070fbe887faa25d2c940eb
MODEL_PARAMS = "model.json"


def load_model_params(path: Path) -> dict:
    """Load model parameters JSON to a dictionary."""
    return json.loads((path / MODEL_PARAMS).read_text())


<<<<<<< HEAD
def default_noflux_platform_to_simulator_channels(
=======
def default_noflux_platform2simulator_channels(
>>>>>>> da0bddaf5ad2c76501070fbe887faa25d2c940eb
    qubits_list: list, couplers_list: list
) -> dict:
    """Returns the default dictionary that maps platform channel names to simulator channel names.
    Args:
        qubits_list (list): List of qubit names to be included in the simulation.
        couplers_list (list): List of coupler names to be included in the simulation.

    Returns:
        dict: Mapping between platform channel names to simulator chanel names.
    """
<<<<<<< HEAD
    return reduce(
        operator.or_,
        [{f"drive-{q}": f"D-{q}", f"readout-{q}": f"R-{q}"} for q in qubits_list]
        + [{f"drive-{c}": f"D-{c}"} for c in couplers_list],
    )
=======
    platform2simulator_channels = {}
    for qubit in qubits_list:
        platform2simulator_channels.update({f"drive-{qubit}": f"D-{qubit}"})
        platform2simulator_channels.update({f"readout-{qubit}": f"R-{qubit}"})
    for coupler in couplers_list:
        platform2simulator_channels.update({f"drive-{coupler}": f"D-{coupler}"})

    return platform2simulator_channels
>>>>>>> da0bddaf5ad2c76501070fbe887faa25d2c940eb
