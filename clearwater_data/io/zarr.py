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
        self.__dataset = xr.open_zarr(self.store_path, consolidated=False)

    def read(self, parameter_name: str) -> DataArrayVariable:
        return DataArrayVariable(self.__dataset[parameter_name].compute())


class ZarrDataStore:
    def __init__(self, **kwargs) -> None:
        self.store_path: Path = kwargs.pop("store_path")
        self.start_date: datetime = kwargs.pop("start_date")
        self.end_date: datetime = kwargs.pop("end_date")
        self.time_step: timedelta = kwargs.pop("time_step")
        self.variables: list[str] = kwargs.pop("variables")

        self.spatial_field: str | None = kwargs.pop("spatial_field", None)
        self.spatial_field_values = kwargs.pop("spatial_field_values", None)

        self._init_zarr_store()

    def _parse_zarr_coordinates(self):
        self.time = pd.date_range(self.start_date, self.end_date, freq=self.time_step)
        dims = ("time",)
        shape = (self.time.shape[0],)
        coords = {"time": self.time}

        if self.spatial_field is not None and self.spatial_field_values is not None:
            dims = ("time", self.spatial_field)
            shape = (self.time.shape[0], len(self.spatial_field_values))
            coords[self.spatial_field] = self.spatial_field_values

        return dims, shape, coords

    def _init_zarr_store(self) -> None:
        dims, shape, coords = self._parse_zarr_coordinates()

        template_dataset = xr.Dataset(
            {v: (dims, da.empty(shape, dtype="float")) for v in self.variables},
            coords=coords,
        )

        # write the template out to generate zarr
        template_dataset.to_zarr(
            self.store_path, mode="w", compute=False, zarr_format=3, consolidated=False
        )

    def write(self, data: ArrayLike, parameter_name: str) -> None:
        data.to_zarr(self.store_path, mode="a", consolidated=False)


class ChunkedZarrDataStore(ZarrDataStore):
    def __init__(self, **kwargs) -> None:
        self.chunk_size: timedelta = kwargs.pop("chunk_size")
        super().__init__(**kwargs)

    def _init_zarr_store(self) -> None:
        dims, shape, coords = self._parse_zarr_coordinates()

        # set chunks
        chunk_length = int(self.chunk_size / self.time_step)
        if self.spatial_field is not None and self.spatial_field_values is not None:
            chunks = (chunk_length, len(self.spatial_field_values))
        else:
            chunks = (chunk_length,)

        template_dataset = xr.Dataset(
            {
                v: (dims, da.empty(shape, dtype="float", chunks=chunks))
                for v in self.variables
            },
            coords=coords,
        )

        # write the template out to generate zarr
        template_dataset.to_zarr(
            self.store_path, mode="w", compute=False, zarr_format=3, consolidated=False
        )

    def write_chunk(
        self,
        data: ArrayLike,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        # parse time indices
        start_index = self.time.get_loc(start_time)
        end_index = self.time.get_loc(end_time)

        # prepare main variable slice; drop auxiliary coordinates
        data_clean = data.drop_vars(
            [c for c in data.coords if c != "time" and c != self.spatial_field]
        )

        data_clean.to_zarr(
            self.store_path,
            # group=parameter_name,
            mode="a",
            region={"time": slice(start_index, end_index + 1)},
            consolidated=False,
        )
