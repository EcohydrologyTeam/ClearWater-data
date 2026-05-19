"""clearwater_data Zarr chunking tests.

B2 (test_chunked_store_spatial_chunk_is_full_extent): ChunkedZarrDataStore
must chunk along time only and keep the full spatial extent in a single
chunk. Pre-fix it used len(self.spatial_field_values) -- the number of
spatial *fields* (~1), not spatial *points* -- so an N-cell store was written
as N one-cell chunks.

B1 (test_chunked_data_source_*): ChunkedZarrDataSource.read_chunk returns only
the requested [start, end] window (bounded memory), satisfies the
ChunkedDataSource protocol, and leaves the inherited eager read() unchanged.

A3 (test_chunked_store_rejects_non_multiple_chunk_size): chunk_size that is
not an exact integer multiple of time_step must fail loudly rather than
silently truncate to a wrong chunk grid.

clearwater_data has no standalone test env; run via a consumer env:
  (cd ../ClearWater-riverine && \
   pixi run -e dev python -m pytest ../ClearWater-data/tests/test_zarr_chunking.py)
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from clearwater_data.io.base import ChunkedDataSource
from clearwater_data.io.zarr import ChunkedZarrDataSource, ChunkedZarrDataStore


def test_chunked_store_spatial_chunk_is_full_extent(tmp_path):
    n_cells = 7
    store = ChunkedZarrDataStore(
        store_path=tmp_path / "store.zarr",
        start_date=datetime(2023, 1, 1, 0, 0),
        end_date=datetime(2023, 1, 1, 1, 0),     # 5 steps at 15 min
        time_step=timedelta(minutes=15),
        chunk_size=timedelta(minutes=30),        # chunk_length = 2
        variables=["concentration"],
        spatial_field=["nface"],
        spatial_field_values=[np.arange(n_cells)],
    )
    ds = xr.open_zarr(store.store_path, consolidated=False)
    chunks = ds["concentration"].chunksizes
    assert chunks["nface"] == (n_cells,), chunks["nface"]
    assert chunks["time"][0] == 2, chunks["time"]


def test_chunked_store_rejects_non_multiple_chunk_size(tmp_path):
    # 25 min is not an integer multiple of a 15 min step -> must raise (A3),
    # not silently truncate int(25/15)=1.
    with pytest.raises(ValueError, match="integer multiple"):
        ChunkedZarrDataStore(
            store_path=tmp_path / "store.zarr",
            start_date=datetime(2023, 1, 1, 0, 0),
            end_date=datetime(2023, 1, 1, 1, 0),
            time_step=timedelta(minutes=15),
            chunk_size=timedelta(minutes=25),
            variables=["concentration"],
            spatial_field=["nface"],
            spatial_field_values=[np.arange(3)],
        )


def _build_store(tmp_path, n_cells, start, end, step):
    store = ChunkedZarrDataStore(
        store_path=tmp_path / "store.zarr",
        start_date=start,
        end_date=end,
        time_step=step,
        chunk_size=timedelta(minutes=30),
        variables=["concentration"],
        spatial_field=["nface"],
        spatial_field_values=[np.arange(n_cells)],
    )
    times = pd.date_range(start, end, freq=step)
    full = xr.DataArray(
        np.arange(len(times) * n_cells, dtype="float").reshape(len(times), n_cells),
        dims=("time", "nface"),
        coords={"time": times, "nface": np.arange(n_cells)},
        name="concentration",
    )
    # Write in two windows via the real chunked-write API.
    store.write_chunk(full.isel(time=slice(0, 3)), "concentration",
                      times[0], times[2])
    store.write_chunk(full.isel(time=slice(3, len(times))), "concentration",
                      times[3], times[-1])
    return store, times, full


def test_chunked_data_source_reads_bounded_window(tmp_path):
    n_cells = 5
    start, end = datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0)
    step = timedelta(minutes=15)
    _store, times, full = _build_store(tmp_path, n_cells, start, end, step)

    src = ChunkedZarrDataSource(store_path=tmp_path / "store.zarr")

    # (b) satisfies the runtime-checkable ChunkedDataSource protocol
    assert isinstance(src, ChunkedDataSource)

    # (a)+(d) read a bounded cross-window slice [times[1], times[3]] (3 stamps)
    da = src.read_chunk("concentration", times[1], times[3]).get()
    assert da.sizes["time"] == 3, da.sizes
    assert da.sizes["nface"] == n_cells, da.sizes
    assert list(pd.to_datetime(da["time"].values)) == list(times[1:4])
    np.testing.assert_array_equal(
        da.values, full.isel(time=slice(1, 4)).values
    )


def test_chunked_data_source_inherited_eager_read_unchanged(tmp_path):
    # Backward-compat: ChunkedZarrDataSource inherits ZarrDataSource.read,
    # which still returns the whole variable (additive change).
    n_cells = 5
    start, end = datetime(2023, 1, 1, 0, 0), datetime(2023, 1, 1, 1, 0)
    step = timedelta(minutes=15)
    _store, times, full = _build_store(tmp_path, n_cells, start, end, step)

    src = ChunkedZarrDataSource(store_path=tmp_path / "store.zarr")
    da = src.read("concentration").get()
    assert da.sizes["time"] == len(times)
    np.testing.assert_array_equal(da.values, full.values)
