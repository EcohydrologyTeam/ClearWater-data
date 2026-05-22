from __future__ import annotations

from abc import ABC, abstractmethod

from clearwater_data import ArrayLike
from datetime import datetime, timedelta


class Variable(ABC):
    """
    Base class for variables.
    """

    @property
    def time_dimension(self) -> str | None:
        """
        Get the time dimension of the variable.
        """
        return None

    @property
    def time_dimension_values(self) -> ArrayLike | None:
        """
        Get the time dimension values of the variable.
        """
        return None

    @property
    def space_dimension(self) -> str | list[str] | None:
        """
        Get the space dimension(s) of the variable.
        """
        return None

    @property
    def space_dimension_values(self) -> ArrayLike | None:
        """
        Get the space dimension(s) values of the variable.
        """
        return None

    @abstractmethod
    def get(self) -> ArrayLike:
        """Return a reference to the variable's value.

        Phase I-5 (2026-05-21): this is the long-standing method
        name. ``get_data`` is the equivalent new-style accessor
        added so the ``VariableRegistry.get`` deprecation message
        (which points users at ``get_variable(key).get_data()``)
        resolves to an actual method on every Variable subclass.
        Default implementation in this base ABC delegates to
        ``get()``; concrete subclasses that override ``get`` will
        be picked up automatically. Subclasses MAY override
        ``get_data`` directly for a more direct path.
        """
        raise NotImplementedError

    def get_data(self) -> ArrayLike:
        """Phase I-5 alias for :meth:`get`.

        See the ``get`` docstring for background. This method exists
        to give the ``VariableRegistry.get`` deprecation message a
        valid migration target without forcing a substrate-wide
        rename of the existing ``get`` method (which would touch
        ~100+ call sites in riverine alone).
        """
        return self.get()

    @abstractmethod
    def get_at_time(
        self,
        time: datetime,
        tolerance: timedelta | None = None,
    ) -> ArrayLike:
        """Return the variable's value at a specific time.

        Phase H-13 (2026-05-21): the optional ``tolerance`` kwarg, when
        non-None, switches the underlying selection to nearest-time
        with the given tolerance. Default ``None`` preserves
        exact-match behaviour for backward compatibility with existing
        callers that have always relied on label-based indexing.

        Subclasses that have no time dimension may ignore the kwarg.
        """
        raise NotImplementedError

    @abstractmethod
    def set(self, value: ArrayLike) -> None:
        """
        Set the variable's value
        """
        raise NotImplementedError

    @abstractmethod
    def set_at_time(self, time: datetime, value: ArrayLike) -> None:
        """
        Set the variable's value at a specific time
        """
        raise NotImplementedError

    @abstractmethod
    def resample(
        self,
        new_time_frequency: timedelta,
        method: str = "linear",
    ) -> None:
        """
        Resample the underlying data to new time frequency
        """
        raise NotImplementedError

    @abstractmethod
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
        raise NotImplementedError

    # TODO: Consider the notion of units in the context of a variable
    # look at `metpy` or `pint` for approaches or inspiration
