from dataclasses import dataclass

from qibolab.channel_config import (
    AcquisitionChannelConfig,
    DCChannelConfig,
    IQChannelConfig,
)
from qibolab.instruments.oscillator import LocalOscillator


@dataclass(frozen=True)
class DCChannelConfigZI(DCChannelConfig):
    """DC channel config using ZI HDAWG."""

    power_range: float
    """Power range in volts.

    Possible values are [0.2 0.4 0.6 0.8 1. 2. 3. 4. 5.].
    """


@dataclass(frozen=True)
class IQChannelConfigZI(IQChannelConfig):
    """IQ channel config for ZI SHF* line instrument."""

    power_range: float
    """Power range in dBm.

    Possible values are [-30. -25. -20. -15. -10. -5. 0. 5. 10.].
    """


class ZIChannel:
    name: str
    device: str
    path: str


class DCChannelZI(ZIChannel):
    config: DCChannelConfigZI


class IQChannelZI(ZIChannel):
    config: IQChannelConfigZI
    lo: LocalOscillator


class AcquisitionChannelZI(ZIChannel):
    config: AcquisitionChannelConfig
