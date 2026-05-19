"""B2 regression: ChunkedZarrDataStore must chunk along time only and keep the
full spatial extent in a single chunk.

Before the fix, ``_init_zarr_store`` used ``len(self.spatial_field_values)``
for the spatial chunk extent. ``spatial_field_values`` is a list of per-field
arrays, so that counted spatial *fields* (~1), not spatial *points* -- an
N-cell store was written as N one-cell chunks, inconsistent with the
coordinate shape built by ``_parse_zarr_coordinates`` (per-field ``len(value)``).

clearwater_data has no standalone test env; run via a consumer env:
  (cd ../ClearWater-riverine && \
   pixi run -e dev python -m pytest ../ClearWater-data/tests/test_zarr_chunking.py)
"""
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from clearwater_data.io.zarr import ChunkedZarrDataStore


def test_chunked_store_spatial_chunk_is_full_extent(tmp_path):
    n_cells = 7
    store = ChunkedZarrDataStore(
        store_path=tmp_path / "store.zarr",
        start_date=datetime(2023, 1, 1, 0, 0),
        end_date=datetime(2023, 1, 1, 1, 0),     # 5 steps at 15 min
        time_step=timedelta(minutes=15),
        chunk_size=timedelta(minutes=30),        # chunk_length = 2
        variables=["concentration"],
        # pass pre-normalized list form (matches _parse_zarr_coordinates'
        # zip(spatial_field, spatial_field_values); avoids the unrelated
        # isinstance(..., Union) normalization path)
        spatial_field=["nface"],
        spatial_field_values=[np.arange(n_cells)],
    )

    ds = xr.open_zarr(store.store_path, consolidated=False)
    chunks = ds["concentration"].chunksizes

    # Spatial dimension: ONE chunk covering all cells (the B2 fix).
    # Pre-fix this was n_cells one-cell chunks => chunks["nface"] == (1,)*7.
    assert chunks["nface"] == (n_cells,), chunks["nface"]

    # Time dimension chunked by chunk_length (= 2): (2, 2, 1) over 5 steps.
    assert chunks["time"][0] == 2, chunks["time"]
