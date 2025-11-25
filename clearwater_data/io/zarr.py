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

        self.spatial_dimension_name: str | None = kwargs.pop("spatial_dimension_name", None)
        self.spatial_dimension_length: str | None = kwargs.pop("spatial_dimension_length", None)

        self._init_zarr_store()

    def __parse_zarr_coordinates(self):
        self.time = pd.date_range(self.start_date, self.end_date, freq=self.time_step)
        dims = ("time",)
        shape = (self.time.shape[0],)
        coords = {"time": self.time}

        if self.spatial_dimension_name and self.spatial_dimension_length:
            dims = ("time", self.spatial_dim_name)
            shape = (self.time.shape[0], self.spatial_dim_length)
            coords[self.spatial_dim_name] = range(self.spatial_dim_length)
        
        return dims, shape, coords


    def __init_zarr_store(self) -> None:
        dims, shape, coords = self.__parse_zarr_coordinates()

        template_dataset = xr.Dataset(
            {
                v: (dims, da.empty(shape, dtype="float"))
                for v in self.variables
            },
            coords=coords,
        )

        # write the template out to generate zarr
        template_dataset.to_zarr(self.store_path, mode="w", compute=False)

    def write(self, data: ArrayLike, parameter_name: str) -> None:
        data.to_zarr(self.store_path, mode="a")


class ChunkedZarrDataStore(ZarrDataStore):
    def __init__(self, **kwargs) -> None:
        self.chunk_size: timedelta = kwargs.pop("chunk_size")
        super().__init__(**kwargs)
    
    def _init_zarr_store(self) -> None:
        dims, shape, coords = self.__parse_zarr_coordinates()

        # set chunks
        if self.spatial_dimension_name and self.spatial_dimension_length:
            chunks = (chunk_length, self.spatial_dimension_length)
        else:
            chunks = (chunk_length,)

        chunk_length = int(self.chunk_size / self.time_step)
        template_dataset = xr.Dataset(
            {
                v: (dims, da.empty(shape, dtype="float", chunks=chunks))
                for v in self.variables
            },
            coords=coords,
        )

        # write the template out to generate zarr
        template_dataset.to_zarr(self.store_path, mode="w", compute=False)

    def write_chunk(
        self,
        data: ArrayLike,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        start_index = self.time.get_loc(start_time)
        end_index = self.time.get_loc(end_time)
        data.to_zarr(
            self.store_path,
            group=parameter_name,
            region={"time": slice(start_index, end_index)}
        )
