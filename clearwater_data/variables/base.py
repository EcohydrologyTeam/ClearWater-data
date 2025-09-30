from abc import ABC, abstractmethod

from clearwater_data import ArrayLike
from datetime import datetime, timedelta


class Variable(ABC):
    """
    Base class for variables.
    """

    @property
    @abstractmethod
    def time_dimension(self) -> str | None:
        """
        Get the time dimension of the variable.
        """
        raise NotImplementedError

    def get(self) -> ArrayLike:
        """
        Get a reference to the variable's value
        """
        raise NotImplementedError

    @abstractmethod
    def get_at_time(self, time: datetime) -> ArrayLike:
        """
        Get a reference to the variable's value at a specific time
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
    # look at metpy
    # look at pint
