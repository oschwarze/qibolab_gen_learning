from typing import Optional

from pydantic import ConfigDict, Field

# TODO: the unused import are there because Qibocal is still importing them from here
# since the export scheme will be reviewed, it should be changed at that time, removing
# the unused ones from here
from .identifier import ChannelId, QubitId, QubitPairId, TransitionId  # noqa
from .serialize import Model


class Qubit(Model):
    """Representation of a physical qubit.

    Contains the channel ids used to control the qubit and is instantiated
    in the function that creates the corresponding
    :class:`qibolab.platforms.platform.Platform`
    """

    model_config = ConfigDict(frozen=False)

    drive: Optional[ChannelId] = None
    """Ouput channel, to drive the qubit state."""
    drive_qudits: dict[TransitionId, ChannelId] = Field(default_factory=dict)
    """Output channels collection, to drive non-qubit transitions."""
    flux: Optional[ChannelId] = None
    """Output channel, to control the qubit flux."""
    probe: Optional[ChannelId] = None
    """Output channel, to probe the resonator."""
    acquisition: Optional[ChannelId] = None
    """Input channel, to acquire the readout results."""

    @property
    def channels(self) -> list[ChannelId]:
        return [
            x
            for x in (
                [getattr(self, ch) for ch in ["probe", "acquisition", "drive", "flux"]]
                + list(self.drive_qudits.values())
            )
            if x is not None
        ]

    @classmethod
    def with_channels(cls, name: QubitId, channels: list[str], **kwargs):
        """Create a qubit with default channel names.

        Default channel names follow the convention:
        '{qubit_name}/{channel_type}'
        """
        return cls(**{ch: f"{name}/{ch}" for ch in channels}, **kwargs)

    @classmethod
    def default(cls, name: QubitId, flux: bool = True, **kwargs):
        """Create a flux tunable qubit with the default channel names.

        Flux tunable qubits have drive, flux, probe and acquisition
        channels.
        """
        channels = ["probe", "acquisition", "drive"]
        if flux:
            channels.append("flux")
        return cls.with_channels(name, channels, **kwargs)


class QubitPair(Model):
    """Represent a two-qubit interaction."""

    drive: Optional[ChannelId] = None
    """Output channel, for cross-resonance driving."""
