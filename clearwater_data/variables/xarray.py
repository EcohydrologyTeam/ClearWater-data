from __future__ import annotations

from clearwater_data.variables.base import Variable
import xarray as xr
from datetime import datetime, timedelta


class DataArrayVariable(Variable):
    def __init__(
        self,
        data_array: xr.DataArray,
        time_dimension: str | None = "time",
        space_dimension: str | list[str] | None = None,
    ) -> None:
        self.data_array = data_array
        self.__time_dimension = time_dimension
        self.__space_dimension = space_dimension

    @property
    def time_dimension(self) -> str | None:
        return self.__time_dimension

    @property
    def space_dimension(self) -> str | list[str] | None:
        return self.__space_dimension

    def get(self) -> xr.DataArray:
        return self.data_array

    def get_at_time(
        self,
        time: datetime,
        tolerance: timedelta | None = None,
    ) -> xr.DataArray:
        """Return the time slice at ``time`` (or the full array if no time dim).

        Phase H-12 (2026-05-21): the guard now checks
        ``self.time_dimension`` consistently rather than the literal
        ``"time"``. Previously the guard tested ``"time" in dims`` while
        the .sel used ``self.time_dimension``: when the underlying
        DataArray used any other time-dim name (e.g., ``"stamp"``), the
        guard was False and the method returned the FULL array instead
        of the requested slice -- a silent correctness failure.

        Phase H-13 (2026-05-21): the optional ``tolerance`` kwarg, when
        non-None, switches the underlying ``.sel`` to ``method='nearest'``
        with the given tolerance. Default ``None`` preserves exact-match
        behaviour so existing callers that rely on byte-identical time
        stamps continue to raise on mismatch.
        """
        if (
            self.time_dimension is not None
            and self.time_dimension in self.data_array.dims
        ):
            if tolerance is not None:
                return self.data_array.sel(
                    {self.time_dimension: time},
                    method="nearest",
                    tolerance=tolerance,
                )
            return self.data_array.sel({self.time_dimension: time})
        return self.data_array

    def set(self, value: xr.DataArray) -> None:
        self.data_array = value

    def set_at_time(self, time: datetime, value: xr.DataArray) -> None:
        """Write ``value`` into the array at ``time``.

        Phase I-4 / N7 (2026-05-21): handle the 1-D (time-only)
        DataArray case explicitly. Previously ``data = self.get_at_time(time)``
        returned a 0-d scalar and ``data[:] = value`` raised
        ``IndexError: too many indices`` on time-only variables.
        Now uses ``.loc`` for the 1-D case (scalar assignment) and
        the slice-assignment for higher-dimensional cases.
        """
        if self.data_array.ndim == 1 and self.time_dimension is not None:
            self.data_array.loc[{self.time_dimension: time}] = value
        else:
            data = self.get_at_time(time)
            data[:] = value

    def resample(
        self,
        new_time_frequency: timedelta,  # Phase I-4 / NIT1: was annotated datetime
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
        """Subset the underlying data to a time range.

        Phase I-4 / N2 (2026-05-21): validate that ``start_time <=
        end_time`` and that the underlying time axis is monotonic
        increasing (xarray's ``.sel(slice(...))`` silently returns
        an empty slice for descending or inverted ranges).
        """
        if (
            start_time is not None
            and end_time is not None
            and start_time > end_time
        ):
            raise ValueError(
                f"subset_time: start_time ({start_time}) > end_time "
                f"({end_time}); slice would be empty."
            )
        if self.time_dimension is not None:
            idx = self.data_array.indexes.get(self.time_dimension)
            if idx is not None and not idx.is_monotonic_increasing:
                raise ValueError(
                    f"subset_time: time axis '{self.time_dimension}' is not "
                    "monotonic increasing; slice would be empty. Sort the "
                    "underlying DataArray first."
                )
        if start_time is not None:
            self.data_array = self.data_array.sel(
                {self.time_dimension: slice(start_time, None)}
            )
        if end_time is not None:
            self.data_array = self.data_array.sel(
                {self.time_dimension: slice(None, end_time)}
            )
