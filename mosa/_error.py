"""Exceptions raised by the MOSA algorithm."""


class MOSAError(Exception):
    """@private
    This class defines exceptions raised by the MOSA algorithm.
    """

    def __init__(self, message: str = "") -> None:
        """Class constructor."""

        self._message = message

    def __str__(self) -> str:
        """Returns the error message."""

        return self._message
