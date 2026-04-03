"""
swot_l3_assembler.py

Utilities for assembling SWOT L3 LR SSH 21-day repeat orbit products
into structured xarray datasets.

Author: <your name>
"""

import glob
import os
import re
from collections import defaultdict
from typing import Iterable, Dict, Optional

import xarray as xr


# --------------------------------------------------
# Internal helpers
# --------------------------------------------------

_PATTERN = re.compile(r'_(\d{3})_(\d{3})_')  # _CYCLE_TRACK_


def _parse_cycle_track(filename: str):
    """Extract cycle and track numbers from SWOT filename."""
    match = _PATTERN.search(os.path.basename(filename))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _clean_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Drop problematic nadir indexing variables."""
    return ds.drop_vars(['i_num_line', 'i_num_pixel'], errors='ignore')


# --------------------------------------------------
# Public API
# --------------------------------------------------

def glob_swot_files(base_path: str) -> list:
    """
    Glob all 21-day orbit files.
    """
    pattern = os.path.join(base_path, "cycle_*", "*.nc")
    return sorted(glob.glob(pattern))


def group_files_by_track(files: Iterable[str],
                         target_tracks: Iterable[int]) -> Dict[int, list]:
    """
    Group files by track number, filtering only desired tracks.
    """
    target_tracks = set(int(t) for t in target_tracks)
    track_files = defaultdict(list)

    for f in files:
        cycle, track = _parse_cycle_track(f)
        if track is None:
            continue
        if track in target_tracks:
            track_files[track].append((cycle, f))

    return track_files


def assemble_track(files_for_track,
                   chunks: Optional[dict] = None) -> xr.Dataset:
    """
    Assemble all cycles for a single track.

    Parameters
    ----------
    files_for_track : list of (cycle, filepath)
    chunks : dict or None
        Optional dask chunking dictionary.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (cycle, num_lines, num_pixels)
    """

    files_for_track = sorted(files_for_track, key=lambda x: x[0])

    cleaned = []

    for cycle, f in files_for_track:
        ds = xr.open_dataset(f, chunks=chunks)
        ds = _clean_dataset(ds)
        ds = ds.expand_dims(cycle=[cycle])
        cleaned.append(ds)

    ds_track = xr.concat(
        cleaned,
        dim='cycle',
        coords='minimal',
        compat='override',
        join='override'
    )

    return ds_track


def assemble_tracks(base_path: str,
                    target_tracks: Iterable[int],
                    merge_tracks: bool = True,
                    chunks: Optional[dict] = None):
    """
    Assemble multiple tracks across all cycles.

    Parameters
    ----------
    base_path : str
        Path to 21day_orbit directory.
    target_tracks : iterable
        Track numbers to assemble.
    merge_tracks : bool
        If True, returns single dataset with track dimension.
        If False, returns dict of datasets keyed by track.
    chunks : dict or None
        Optional dask chunking.

    Returns
    -------
    xr.Dataset or dict
    """

    files = glob_swot_files(base_path)
    track_groups = group_files_by_track(files, target_tracks)

    track_datasets = {}

    for track, files_for_track in track_groups.items():
        ds_track = assemble_track(files_for_track, chunks=chunks)
        ds_track = ds_track.expand_dims(track=[track])
        track_datasets[track] = ds_track

    if not merge_tracks:
        return track_datasets

    all_cycles = sorted(
        set(
            c
            for ds in track_datasets.values()
            for c in ds.cycle.values
        )
    )

    for track in track_datasets:
        track_datasets[track] = track_datasets[track].reindex(
            cycle=all_cycles
        )

    ds_all = xr.concat(
        track_datasets.values(),
        dim='track',
        coords='minimal',
        compat='override',
        join='override'
    )

    return ds_all


def assemble_by_pass_direction(base_path: str,
                               descending_tracks: Iterable[int],
                               ascending_tracks: Iterable[int],
                               chunks: Optional[dict] = None):
    """
    Assemble descending and ascending tracks separately.

    Parameters
    ----------
    base_path : str
        Path to 21day_orbit directory.
    descending_tracks : iterable
        Track numbers belonging to descending passes.
    ascending_tracks : iterable
        Track numbers belonging to ascending passes.
    chunks : dict or None
        Optional dask chunking.

    Returns
    -------
    ds_desc : xr.Dataset
        Merged descending dataset
    ds_asc : xr.Dataset
        Merged ascending dataset
    """

    files = glob_swot_files(base_path)

    # -----------------------------
    # Group files separately
    # -----------------------------
    desc_groups = group_files_by_track(files, descending_tracks)
    asc_groups = group_files_by_track(files, ascending_tracks)

    if not desc_groups:
        raise RuntimeError("No descending files found.")
    if not asc_groups:
        raise RuntimeError("No ascending files found.")

    # -----------------------------
    # Assemble individual tracks
    # -----------------------------
    desc_datasets = {}
    asc_datasets = {}

    for track, files_for_track in desc_groups.items():
        ds_track = assemble_track(files_for_track, chunks=chunks)
        ds_track = ds_track.expand_dims(track=[track])
        desc_datasets[track] = ds_track

    for track, files_for_track in asc_groups.items():
        ds_track = assemble_track(files_for_track, chunks=chunks)
        ds_track = ds_track.expand_dims(track=[track])
        asc_datasets[track] = ds_track

    # -----------------------------
    # Harmonize cycles within each group
    # -----------------------------
    def _harmonize_cycles(track_dict):
        all_cycles = sorted(
            set(
                c
                for ds in track_dict.values()
                for c in ds.cycle.values
            )
        )

        for track in track_dict:
            track_dict[track] = track_dict[track].reindex(
                cycle=all_cycles
            )

        return track_dict

    desc_datasets = _harmonize_cycles(desc_datasets)
    asc_datasets = _harmonize_cycles(asc_datasets)

    # -----------------------------
    # Merge safely (geometry-consistent only)
    # -----------------------------
    # ds_desc = xr.concat(
    #     desc_datasets.values(),
    #     dim='track',
    #     coords='minimal',
    #     compat='override',
    #     join='override'
    # )

    # ds_asc = xr.concat(
    #     asc_datasets.values(),
    #     dim='track',
    #     coords='minimal',
    #     compat='override',
    #     join='override'
    # )

    ds_desc = xr.concat(
        desc_datasets.values(),
        dim='track',
        coords='different', 
        compat='no_conflicts',
        join='outer'
    )

    ds_asc = xr.concat(
        asc_datasets.values(),
        dim='track',
        coords='different', 
        compat='no_conflicts',
        join='outer'
    )

    return ds_desc, ds_asc
