"""Locks the clearwater_data.io public API.

io/__init__.py was empty, so concrete sources/stores, provider protocols, and
path helpers had no short import path. These tests assert the short path
exposes the stable surface and that the existing submodule paths still work
(the change is additive).
"""
import clearwater_data.io as cio

EXPECTED = (
    "DataSource",
    "ChunkedDataSource",
    "DataStore",
    "ChunkedDataStore",
    "DataProvider",
    "ChunkedDataProvider",
    "ZarrDataSource",
    "ChunkedZarrDataSource",
    "ZarrDataStore",
    "ChunkedZarrDataStore",
    "CSVDataSource",
    "FloatDataSource",
    "resolve_path",
    "validate_path",
)


def test_io_short_import_path_exposes_public_api():
    from clearwater_data.io import (  # noqa: F401
        ChunkedDataProvider,
        ChunkedDataSource,
        ChunkedDataStore,
        ChunkedZarrDataSource,
        ChunkedZarrDataStore,
        CSVDataSource,
        DataProvider,
        DataSource,
        DataStore,
        FloatDataSource,
        ZarrDataSource,
        ZarrDataStore,
        resolve_path,
        validate_path,
    )
    for name in EXPECTED:
        assert name in cio.__all__, f"{name} missing from clearwater_data.io.__all__"
        assert hasattr(cio, name), f"{name} not importable from clearwater_data.io"


def test_io_submodule_paths_still_work():
    # Backward-compat: existing long paths unaffected by the additive __init__.
    from clearwater_data.io.base import ChunkedDataSource, DataSource  # noqa: F401
    from clearwater_data.io.zarr import (  # noqa: F401
        ChunkedZarrDataSource,
        ZarrDataSource,
        ZarrDataStore,
    )
