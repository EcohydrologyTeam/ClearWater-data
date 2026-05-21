from __future__ import annotations

from clearwater_data.variables.base import Variable
from datetime import datetime, timedelta


class FloatVariable(Variable):
    """
    A variable that stores a single float value.
    """

    time_dimension = None
    space_dimension = None

    def __init__(self, value: float):
        self.value = value

    def get(self) -> float:
        """
        Get a reference to the variable's value
        """
        return self.value

    def get_at_time(
        self,
        time: datetime,
        tolerance: timedelta | None = None,
    ) -> float:
        """Return the scalar value (time-independent).

        Phase H-13 (2026-05-21): ``tolerance`` is accepted for
        signature parity with the base class but has no effect here
        (the scalar is time-independent by definition).
        """
        return self.get()

    def set(self, value: float) -> None:
        """
        Set the variable's value
        """
        self.value = value

    def set_at_time(self, time: datetime, value: float) -> None:
        """Reject per-time set: FloatVariable is time-independent.

        Phase H-15 (2026-05-21): previously this method silently
        discarded the ``time`` argument and mutated the global scalar
        via ``self.set(value)``. A caller building a per-time sequence
        of FloatVariable updates would see only the LAST value persist,
        with no diagnostic. Raise loudly instead so the contract is
        explicit. Callers who genuinely want to update the scalar
        globally should call ``set(value)`` directly.
        """
        raise NotImplementedError(
            "FloatVariable.set_at_time is unsupported because FloatVariable "
            "stores a single time-independent scalar. The 'time' argument "
            "would be silently ignored if accepted, masking per-time-set bugs. "
            "Call set(value) directly to update the scalar globally."
        )

    def resample(
        self,
        new_time_frequency: timedelta,
        method: str = "linear",
    ) -> None:
        """
        Resample the underlying data to new time frequency
        """
        # single floating value is time independent, no need to resample
        # but we need to implement the method to satisfy the interface
        return None

    def subset_time(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        """
        Subset the underlying data to a specific time range.
            This is primarily useful for aligning variables to temporal ranges.
        Note: this will modify the variable's data inplace
            avoid using this for selecting a subset of timesteps.
        """
        # single floating value is time independent, no need to subset
        # but we need to implement the method to satisfy the interface
        return None
