from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Optional

from qibolab.unrolling import Bounds

InstrumentId = str


@dataclass
class InstrumentSettings:
    """Container of settings that are dumped in the platform runcard json."""

    def dump(self):
        """Dictionary containing the settings.

        Useful when dumping the instruments to the runcard JSON.
        """
        return asdict(self)


class Instrument(ABC):
    """Parent class for all the instruments connected via TCPIP.

    Args:
        name (str): Instrument name.
        address (str): Instrument network address.
    """

    def __init__(self, name, address):
        self.name: InstrumentId = name
        self.address: str = address
        self.is_connected: bool = False
        self.settings: Optional[InstrumentSettings] = None

    @property
    def signature(self):
        return f"{type(self).__name__}@{self.address}"

    @abstractmethod
    def connect(self):
        """Establish connection to the physical instrument."""

    @abstractmethod
    def disconnect(self):
        """Close connection to the physical instrument."""

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Set instrument settings."""


class Controller(Instrument):
    """Instrument that can play pulses (using waveform generator)."""

    def __init__(self, name, address):
        super().__init__(name, address)
        self._ports = {}
        self.bounds: Bounds = Bounds(0, 0, 0)
        """Estimated limitations of the device memory."""

    def setup(self, bounds):
        """Set unrolling batch bounds."""
        self.bounds = Bounds(**bounds)

    def dump(self):
        """Dump unrolling batch bounds."""
        return {"bounds": asdict(self.bounds)}

    @property
    @abstractmethod
    def sampling_rate(self):
        """Sampling rate of control electronics in giga samples per second
        (GSps)."""

    @abstractmethod
    def play(self, *args, **kwargs):
        """Play a pulse sequence and retrieve feedback.

        Returns:
            (Dict[ResultType]) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """

    @abstractmethod
    def sweep(self, *args, **kwargs):
        """Play a pulse sequence while sweeping one or more parameters.

        Returns:
            (dict) mapping the serial of the readout pulses used to
            the acquired :class:`qibolab.result.ExecutionResults` object.
        """
