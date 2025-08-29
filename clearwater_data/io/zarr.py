from clearwater_data.custom_types import ArrayLike
from datetime import datetime, timedelta
from pathlib import Path
import xarray as xr
import dask.array as da
import pandas as pd
from clearwater_data.variables.xarray import DataArrayVariable


class ZarrDataSource:
    def __init__(self, **kwargs) -> None:
        self.store_path: Path = kwargs.pop("store_path")
        self.__dataset = xr.open_zarr(self.store_path)

    def read(self, parameter_name: str) -> DataArrayVariable:
        return DataArrayVariable(self.__dataset[parameter_name].compute())


class ZarrDataStore:
    def __init__(self, **kwargs) -> None:
        self.store_path: Path = kwargs.pop("store_path")
        self.start_date: datetime = kwargs.pop("start_date")
        self.end_date: datetime = kwargs.pop("end_date")
        self.time_step: timedelta = kwargs.pop("time_step")
        self.variables: list[str] = kwargs.pop("variables")

        self.__init_zarr_store()

    def __init_zarr_store(self) -> None:
        time = pd.date_range(self.start_date, self.end_date, freq=self.time_step)
        template_dataset = xr.Dataset(
            {
                v: (("time"), da.empty((time.shape[0]), dtype="float"))
                for v in self.variables
            },
            coords={"time": time},
        )

        # write the template out to generate zarr
        template_dataset.to_zarr(self.store_path, mode="w", compute=False)

    def write(self, data: ArrayLike, parameter_name: str) -> None:
        data.to_zarr(self.store_path, mode="a")


class ChunkedZarrDataStore:
    def __init__(self, **kwargs) -> None:
        self.store_path = kwargs.pop("store_path")

    def write_chunk(
        self,
        data: ArrayLike,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        data.to_zarr(self.store_path, mode="a")
