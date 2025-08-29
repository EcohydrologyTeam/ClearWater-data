from clearwater_data.io.base import DataSource
from clearwater_data.custom_types import ArrayLike
from clearwater_data.variables import FloatVariable


class FloatDataSource(DataSource):
    def __init__(self, **kwargs) -> None:
        self.value = kwargs.pop("value")

    def read(self, parameter_name: str) -> ArrayLike:
        return FloatVariable(self.value)
