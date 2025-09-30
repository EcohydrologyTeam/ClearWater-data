from clearwater_data.variables.base import Variable
import xarray as xr
from datetime import datetime


class DataArrayVariable(Variable):
    def __init__(
        self, data_array: xr.DataArray, time_dimension: str | None = "time"
    ) -> None:
        self.data_array = data_array
        self.__time_dimension = time_dimension

    @property
    def time_dimension(self) -> str | None:
        return self.__time_dimension

    def get(self) -> xr.DataArray:
        return self.data_array

    def get_at_time(self, time: datetime) -> xr.DataArray:
        # if data a has time dimension, return the value at that time
        if "time" in self.data_array.dims:
            return self.data_array.sel({self.time_dimension: time})
        # otherwise return the value
        return self.data_array

    def resample(
        self,
        new_time_frequency: datetime,
        method: str = "linear",
    ) -> None:
        if self.time_dimension is None:
            raise ValueError("Cannot resample a variable with no time dimension")

        self.data_array = self.data_array.resample(
            {self.time_dimension: new_time_frequency}
        ).interpolate(method)

    def subset_time(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> None:
        if start_time is not None:
            self.data_array = self.data_array.sel(
                {self.time_dimension: slice(start_time, None)}
            )
        if end_time is not None:
            self.data_array = self.data_array.sel(
                {self.time_dimension: slice(None, end_time)}
            )
