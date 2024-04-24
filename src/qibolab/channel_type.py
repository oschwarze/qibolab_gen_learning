from typing import Protocol

from .channel_config import ChannelConfig


class Channel(Protocol):
    """This is a class intended to be used internally in qibolab for type
    annotations only.

    Channel objects should have at least two properties as defined
    below. We do not enforce a certain abstract or non-abstract parent
    class for channels.
    """

    name: str
    config: ChannelConfig
