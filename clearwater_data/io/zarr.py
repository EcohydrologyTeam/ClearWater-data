from clearwater_data.custom_types import ArrayLike
from datetime import datetime, timedelta
from pathlib import Path
import xarray as xr
import dask.array as da
import pandas as pd
from clearwater_data.variables.xarray import DataArrayVariable


from logging import getLogger

LOGGER = getLogger(__name__)


class ZarrDataSource:
    def __init__(self, **kwargs) -> None:
        self.store_path: Path = kwargs.pop("store_path")
        self.__dataset = xr.open_zarr(self.store_path, consolidated=False)

    def read(self, parameter_name: str) -> DataArrayVariable:
        return DataArrayVariable(self.__dataset[parameter_name].compute())


class ChunkedZarrDataSource(ZarrDataSource):
    """Chunked (windowed) reader satisfying the ChunkedDataSource protocol.

    Lazily opens the Zarr store and materializes ONLY the requested
    [start_time, end_time] window, so resident memory is bounded by the
    window rather than the whole variable. Additive: ``ZarrDataSource`` and
    its eager ``read`` (inherited unchanged) are not modified, so existing
    consumers of ``ZarrDataSource``/``read`` are unaffected (B1).
    """

    def read_chunk(
        self,
        parameter_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> DataArrayVariable:
        """Read the [start_time, end_time] window of a stored variable.

        Phase I-4 / N1 (2026-05-21): validate the request bounds.
        Previously a request inverted ``end_time < start_time`` or
        entirely outside the store's time extent returned a
        zero-length DataArray silently; downstream consumers then
        hit confusing "empty array" mystery failures. Raise loudly
        with a clear message naming the store's actual range.
        """
        if end_time < start_time:
            raise ValueError(
                f"read_chunk: end_time ({end_time}) < start_time "
                f"({start_time}); window is empty."
            )
        # Lazy open (metadata only); materialize ONLY the requested window
        # so resident memory is bounded by the window, not the whole array.
        dataset = xr.open_zarr(self.store_path, consolidated=False)
        store_t0 = pd.Timestamp(dataset.time.values[0])
        store_tN = pd.Timestamp(dataset.time.values[-1])
        if end_time < store_t0 or start_time > store_tN:
            raise ValueError(
                f"read_chunk: window [{start_time}, {end_time}] does not "
                f"overlap the store's time extent [{store_t0}, {store_tN}]."
            )
        window = dataset[parameter_name].sel(time=slice(start_time, end_time))
        return DataArrayVariable(window.compute())


class ZarrDataStore:
    def __init__(self, **kwargs) -> None:
        self.store_path: Path = kwargs.pop("store_path")
        self.start_date: datetime = kwargs.pop("start_date")
        self.end_date: datetime = kwargs.pop("end_date")
        self.time_step: timedelta = kwargs.pop("time_step")
        self.variables: list[str] = kwargs.pop("variables")
        # Phase G-4 (2026-05-21): accept an explicit time-coord vector
        # rather than synthesizing a uniform grid from
        # (start_date, end_date, time_step). When supplied, the
        # actual stamps the caller intends to write are used as the
        # template's time axis. This is required when the upstream
        # data (e.g., HEC-RAS HDF outputs) carries non-uniform time
        # stamps; otherwise the chunk write attempts to land on stamps
        # that do not exist in the uniform template and either raises
        # ``KeyError`` (newer xarray) or silently leaves slots NaN
        # (older xarray). Default ``None`` preserves the original
        # behavior of ``pd.date_range(start, end, freq=time_step)``.
        self.time_coord: pd.DatetimeIndex | None = kwargs.pop("time_coord", None)
        if self.time_coord is not None:
            self.time_coord = pd.DatetimeIndex(self.time_coord)
        # When False, skip the mode="w" template init so an existing store
        # at ``store_path`` is preserved. Required for the riverine
        # checkpoint/resume path (Phase-C C3b): a resumed run must continue
        # writing into the same pre-allocated store rather than clobbering
        # the chunks already written by the original run. Default True keeps
        # existing callers unchanged.
        init_template: bool = kwargs.pop("init_template", True)

        # TODO: we should rename these to space to be consistent with variable definitions
        # add in deprecation warning for the old names
        self.spatial_field: str | list[str] | None = kwargs.pop("spatial_field", None)
        self.spatial_field_values: ArrayLike | list[ArrayLike] | None = kwargs.pop(
            "spatial_field_values", None
        )

        if isinstance(self.spatial_field, str) and isinstance(
            self.spatial_field_values, ArrayLike
        ):
            self.spatial_field = [self.spatial_field]
            self.spatial_field_values = [self.spatial_field_values]

        # Phase I-4 / N6 (2026-05-21): reject typos in optional
        # kwargs. Previously a caller's typo'd kwarg (e.g.,
        # ``inti_template=True``) was silently dropped because
        # ``**kwargs.pop(...)`` consumed only the names this class
        # knows about. Now any leftover kwargs raise TypeError with
        # a list of the unexpected names so the user catches the
        # typo at construction time.
        if kwargs:
            raise TypeError(
                f"ZarrDataStore got unexpected kwargs: {sorted(kwargs)}. "
                "Check for typos or extra arguments."
            )

        if init_template:
            self._init_zarr_store()

    def _parse_zarr_coordinates(self):
        # Phase G-4 (2026-05-21): prefer the caller-supplied
        # time_coord (actual stamps) over the synthesized
        # uniform-grid date_range so chunk writes against
        # non-uniform RAS time series land on stamps the template
        # actually contains.
        if self.time_coord is not None:
            self.time = self.time_coord
        else:
            self.time = pd.date_range(
                self.start_date, self.end_date, freq=self.time_step,
            )
        dims = ("time",)
        shape = (self.time.shape[0],)
        coords = {"time": self.time}

        if self.spatial_field is not None and self.spatial_field_values is not None:
            # TODO: ask Sarah if this needs to be a tuple?
            for name, value in zip(self.spatial_field, self.spatial_field_values):
                dims = (*dims, name)
                shape = (*shape, len(value))
                coords[name] = value

        return dims, shape, coords

    def _init_zarr_store(self) -> None:
        dims, shape, coords = self._parse_zarr_coordinates()

        # needs to be updated to support format
        template_dataset = xr.Dataset(
            {v: (dims, da.empty(shape, dtype="float")) for v in self.variables},
            coords=coords,
        )

        # write the template out to generate zarr
        template_dataset.to_zarr(
            self.store_path, mode="w", compute=False, zarr_format=3, consolidated=False
        )

    def write(self, data: ArrayLike, parameter_name: str) -> None:
        """Append ``data`` to the pre-allocated template under ``parameter_name``.

        Phase H-11 (2026-05-21): if ``data.name != parameter_name``,
        rename the DataArray before writing so the value lands in the
        intended slot of the template. Previously ``parameter_name``
        was accepted but ignored; the target variable was whatever
        ``data.name`` happened to carry, so a caller's safe-looking
        ``store.write(da, 'concentration')`` could land the values
        under a different name with no diagnostic.
        """
        if hasattr(data, "name") and data.name != parameter_name:
            data = data.rename(parameter_name)
        data.to_zarr(self.store_path, mode="a", consolidated=False, compute=True)


class ChunkedZarrDataStore(ZarrDataStore):
    def __init__(self, **kwargs) -> None:
        self.chunk_size: timedelta = kwargs.pop("chunk_size")
        super().__init__(**kwargs)

    def _init_zarr_store(self) -> None:
        dims, shape, coords = self._parse_zarr_coordinates()

        # set chunks: chunk along time only; keep the full spatial extent in
        # one chunk. `shape` == (n_time, *spatial_extents) from
        # _parse_zarr_coordinates; the previous len(self.spatial_field_values)
        # counted spatial *fields* (~1), not spatial *points* (B2).
        #
        # chunk_size must be an exact integer multiple of time_step; otherwise
        # int(chunk_size / time_step) would silently truncate to a wrong chunk
        # grid. Use exact timedelta arithmetic and fail loudly instead (A3).
        if self.chunk_size % self.time_step != timedelta(0):
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be an integer multiple "
                f"of time_step ({self.time_step})."
            )
        chunk_length = self.chunk_size // self.time_step
        chunks = (chunk_length, *shape[1:])

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
        LOGGER.debug(
            f"Writing chunk for {parameter_name} from {start_time} to {end_time} to zarr store at {self.store_path}"
        )

        # prepare main variable slice; drop auxiliary coordinates.
        #
        # Phase G-5 (2026-05-21): use set-membership against the
        # normalized spatial_field list rather than ``c != self.spatial_field``
        # which compares a string coord-name to a list and is always
        # True. The previous form stripped the spatial coord on every
        # chunk write; latent because tests did not attach auxiliary
        # coords. ``self.spatial_field`` is promoted to a list at
        # ``__init__`` time when it arrives as a string.
        spatial_fields = (
            self.spatial_field if isinstance(self.spatial_field, list)
            else ([self.spatial_field] if self.spatial_field is not None else [])
        )
        keep = {"time", *spatial_fields}
        data_clean = data.drop_vars(
            [c for c in data.coords if c not in keep]
        )

        # Phase J+1 (2026-05-23): rename to parameter_name before writing,
        # mirroring the same protection ``write()`` got in Phase H-11. The
        # template was pre-allocated under ``parameter_name``; if the
        # incoming ``data.name`` was inherited from the IC/BC CSV's
        # ``value_field`` (e.g. ``"Concentration"``), ``to_zarr`` writes
        # to a different variable than the template, leaving the
        # pre-allocated slot untouched AND creating a stray short-time
        # variable that breaks subsequent chunk writes on the second
        # constituent ("variable 'Concentration' already exists with
        # different dimension sizes"). Visible only in chunked mode
        # because non-chunked uses ``write()`` which already has the
        # protection.
        if hasattr(data_clean, "name") and data_clean.name != parameter_name:
            data_clean = data_clean.rename(parameter_name)

        data_clean.to_zarr(
            self.store_path,
            # group=parameter_name,
            mode="a",
            # region={"time": slice(start_index, end_index + 1)},
            consolidated=False,
            region="auto",
            compute=True,
        )
