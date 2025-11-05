from clearwater_data.custom_types import ArrayLike
from clearwater_data.variables import DataArrayVariable
from pathlib import Path
import pandas as pd


class CSVDataSource:
    def __init__(self, **kwargs) -> None:
        self.file_path: Path = kwargs.pop("file_path")
        self.time_field: str = kwargs.pop("time_field", None)
        self.spatial_field: str = kwargs.pop("spatial_field", None)
        # self.interpolation_method = kwargs.pop("interpolation_method", "linear")
        self.__data: ArrayLike | None = None

    def read(self, parameter_name: str) -> DataArrayVariable:
        # load data if we don't have it cached
        if self.__data is None:
            df = pd.read_csv(self.file_path)

            df = df.rename(columns={self.time_field: "time"})
            df["time"] = pd.to_datetime(df["time"])
            
            # If spatial field is defined, set a multi-index for [time, spatial]
            if self.spatial_field is not None and self.spatial_field in df.columns:
                df = df.set_index(["time", self.spatial_field])
            else:
                df = df.set_index("time")

            self.__data = df.to_xarray()

        return DataArrayVariable(self.__data[parameter_name], "time")
