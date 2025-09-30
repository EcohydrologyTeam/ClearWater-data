from clearwater_data.variables.base import Variable
from datetime import datetime, timedelta


class FloatVariable(Variable):
    """
    A variable that stores a single float value.
    """

    time_dimension = None

    def __init__(self, value: float):
        self.value = value

    def get(self) -> float:
        """
        Get a reference to the variable's value
        """
        return self.value

    def get_at_time(self, time: datetime) -> float:
        """
        Get a reference to the variable's value at a specific time
        """
        # single floating value is time independent
        return self.get()

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
