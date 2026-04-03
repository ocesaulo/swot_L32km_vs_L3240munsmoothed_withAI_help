"""Utilities for reading SWOT L3 swath files and finding California coverage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re

# Some local environments in this repo have mixed HDF5 libs; suppress hard abort.
os.environ.setdefault("HDF5_DISABLE_VERSION_CHECK", "2")

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class BoundingBox:
    """Simple lat/lon bounding box in degrees East convention."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


# Approximate California coast window in 0..360 longitude.
CALIFORNIA_COAST_BBOX = BoundingBox(
    lon_min=236.0,  # -124 deg
    lon_max=246.0,  # -114 deg
    lat_min=31.0,
    lat_max=43.0,
)


def _parse_track_from_filename(path: Path) -> int | None:
    # Common SWOT naming includes ..._<cycle>_<track>_<timestamp>_... .nc
    match = re.search(r"_(\d{3})_(\d{3})_\d{8}T", path.name)
    if match:
        return int(match.group(2))

    # Fallback for older variants where track may still precede timestamp directly.
    match = re.search(r"_(\d{3})_\d{8}T", path.name)
    if match:
        return int(match.group(1))
    return None


def _find_var_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str:
    """Return the first existing variable/coord name from candidates."""
    for name in candidates:
        if name in ds.variables or name in ds.coords:
            return name
    raise KeyError(f"None of the candidate names found: {candidates}")


def _decode_scaled_array(da: xr.DataArray) -> np.ndarray:
    """Decode a variable with optional scale_factor/add_offset/fill value."""
    arr_raw = da.values
    arr = da.astype("float64").values

    scale = float(da.attrs.get("scale_factor", 1.0))
    offset = float(da.attrs.get("add_offset", 0.0))
    arr = arr * scale + offset

    fill = da.attrs.get("_FillValue", None)
    if fill is not None:
        arr[arr_raw == fill] = np.nan

    return arr


def _to_360(lon: np.ndarray) -> np.ndarray:
    """Normalize longitudes to [0, 360)."""
    out = np.asarray(lon, dtype="float64").copy()
    mask = np.isfinite(out)
    out[mask] = np.mod(out[mask], 360.0)
    return out


def list_cycle_files(base_path: str | Path, cycle: int) -> list[Path]:
    """Return SWOT L3 files for one cycle, sorted by track index in filename."""
    base = Path(base_path)
    cycle_dir = base / f"cycle_{cycle:03d}"
    if not cycle_dir.exists():
        raise FileNotFoundError(f"Missing cycle directory: {cycle_dir}")

    # Accept both Expert and Unsmoothed products, and future variants.
    files = sorted(cycle_dir.glob("SWOT_L3_LR_SSH_*.nc"))

    # Fallback to any NetCDF if prefix differs in future deliveries.
    if not files:
        files = sorted(cycle_dir.glob("*.nc"))

    if not files:
        raise FileNotFoundError(f"No SWOT files found under {cycle_dir}")
    return sorted(files, key=lambda p: (_parse_track_from_filename(p) is None, _parse_track_from_filename(p), p.name))


def open_swot_file(path: str | Path, chunks: dict[str, int] | None = None) -> xr.Dataset:
    """Open a SWOT file with scale/offset decoding and no time decoding."""
    return xr.open_dataset(
        path,
        chunks=chunks,
        decode_times=False,
        decode_cf=False,
        mask_and_scale=False,
    )


def file_intersects_bbox(path: str | Path, bbox: BoundingBox = CALIFORNIA_COAST_BBOX) -> dict:
    """Check if any swath sample in a SWOT file intersects the requested box."""
    with open_swot_file(path) as ds:
        lon_name = _find_var_name(ds, ("longitude", "lon", "LON", "Longitude"))
        lat_name = _find_var_name(ds, ("latitude", "lat", "LAT", "Latitude"))
        lon = _decode_scaled_array(ds[lon_name])
        lat = _decode_scaled_array(ds[lat_name])

    lon_360 = _to_360(lon)

    valid = np.isfinite(lon_360) & np.isfinite(lat)
    in_box = (
        valid
        & (lon_360 >= bbox.lon_min)
        & (lon_360 <= bbox.lon_max)
        & (lat >= bbox.lat_min)
        & (lat <= bbox.lat_max)
    )

    samples = int(in_box.sum())
    track = _parse_track_from_filename(Path(path))
    return {
        "file": str(path),
        "track": track,
        "intersects": samples > 0,
        "sample_count": samples,
    }


def california_passes_for_cycle(
    cycle: int,
    swot_root: str | Path = "data/swot/1day_orbit",
    bbox: BoundingBox = CALIFORNIA_COAST_BBOX,
) -> list[dict]:
    """Return pass summaries for one cycle, sorted by track number."""
    files = list_cycle_files(swot_root, cycle)
    results = [file_intersects_bbox(path, bbox=bbox) for path in files]
    return sorted(results, key=lambda x: (x["track"] is None, x["track"]))
