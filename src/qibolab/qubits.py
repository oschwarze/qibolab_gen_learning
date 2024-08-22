from collections.abc import Iterable
from typing import Optional

from pydantic import ConfigDict

from .components.channels import Channel
from .identifier import ChannelId, ChannelType, QubitId
from .serialize import Model


class Qubit(Model):
    """Representation of a physical qubit.

    Qubit objects are instantiated by :class:`qibolab.platforms.platform.Platform`
    but they are passed to instrument designs in order to play pulses.

    Args:
        name (int, str): Qubit number or name.
        readout (:class:`qibolab.platforms.utils.Channel`): Channel used to
            readout pulses to the qubit.
        drive (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send drive pulses to the qubit.
        flux (:class:`qibolab.platforms.utils.Channel`): Channel used to
            send flux pulses to the qubit.
    """

    model_config = ConfigDict(frozen=False)

    name: QubitId

    probe: Optional[ChannelId] = None
    acquisition: Optional[ChannelId] = None
    drive: Optional[ChannelId] = None
    drive_qudits: Optional[dict[str, ChannelId]] = None
    flux: Optional[ChannelId] = None

    @property
    def channels(self) -> Iterable[Channel]:
        for ct in ChannelType:
            channel = getattr(self, ct.value)
            if channel is not None:
                yield channel

    @property
    def mixer_frequencies(self):
        """Get local oscillator and intermediate frequencies of native gates.

        Assumes RF = LO + IF.
        """
        freqs = {}
        for name in self.native_gates.model_fields:
            native = getattr(self.native_gates, name)
            if native is not None:
                channel_type = native.pulse_type.name.lower()
                _lo = getattr(self, channel_type).lo_frequency
                _if = native.frequency - _lo
                freqs[name] = _lo, _if
        return freqs
