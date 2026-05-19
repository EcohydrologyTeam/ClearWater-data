"""Public I/O API for clearwater_data.

Short, stable import path for the concrete data sources/stores, the provider
protocols, and the path helpers, so consumers (clearwater_riverine,
clearwater_modules, ESM) can ``from clearwater_data.io import X`` instead of
reaching into submodules. The submodule import paths
(``clearwater_data.io.zarr`` etc.) remain valid -- this is additive.
"""
from clearwater_data.io.base import (
    ChunkedDataProvider,
    ChunkedDataSource,
    ChunkedDataStore,
    DataProvider,
    DataSource,
    DataStore,
)
from clearwater_data.io.csv import CSVDataSource
from clearwater_data.io.float import FloatDataSource
from clearwater_data.io.pathing import resolve_path, validate_path
from clearwater_data.io.zarr import (
    ChunkedZarrDataSource,
    ChunkedZarrDataStore,
    ZarrDataSource,
    ZarrDataStore,
)

__all__ = [
    # provider protocols / ABCs
    "DataSource",
    "ChunkedDataSource",
    "DataStore",
    "ChunkedDataStore",
    "DataProvider",
    "ChunkedDataProvider",
    # zarr sources / stores
    "ZarrDataSource",
    "ChunkedZarrDataSource",
    "ZarrDataStore",
    "ChunkedZarrDataStore",
    # other concrete sources
    "CSVDataSource",
    "FloatDataSource",
    # path helpers
    "resolve_path",
    "validate_path",
]
