> Disclaimer: This notebook and its supporting workflow were developed with significant input from AI assistance. Review the methods, assumptions, and results carefully before using them for scientific interpretation.

# Cycle 503 California Pass Reader

This README summarizes only the notebook `cycle503_california_pass_reader.ipynb`.

## Purpose

The notebook analyzes SWOT L3 LR SSH Expert data over the California coast, starting with cycle 503 and then expanding into multi-cycle spectral comparisons. Its main goal is to identify which SWOT passes intersect the California coastal region, inspect the dominant pass geometry and data quality, and compare along-track SSH spectra across products and swath subsets.

## Notebook Workflow

### Step 1: Find cycle 503 passes intersecting California

The notebook uses helper functions from `swot_cycle_reader.py` to:

- enumerate all files for cycle 503,
- test which passes intersect a California coastal bounding box,
- build a table of intersecting tracks and sample counts.

### Step 2: Inspect the dominant California pass

The strongest California-coverage pass is opened and examined in more detail. The notebook:

- decodes latitude and longitude,
- subsets the swath to the California bounding box,
- maps the selected track centerline,
- estimates the median along-track spacing,
- computes basic quality-filtered SSHA statistics over the in-box samples.

### Step 3: Compute a center-swath PSD for cycle 503

Using `swot_l3_assembler.py` and a notebook-local adaptation of the `freq_spec` spectrum routine, the notebook:

- assembles the selected track,
- extracts two center-swath pixels,
- finds the longest shared finite segment,
- computes a one-cycle SSH power spectral density,
- plots the center-swath spectrum on log-log axes.

### Step 4: Extend track 013 across all available cycles

The notebook then broadens the analysis to track 013 across all available unsmoothed cycles. It:

- extracts center-swath profiles for each cycle,
- trims them to a common usable segment,
- computes an ensemble-average spectrum,
- estimates 95% confidence limits,
- plots the all-cycle average PSD.

### Step 5: Compare against L3 2 km smoothed data

For cycles available in both archives, the notebook compares the unsmoothed product against the 2 km smoothed L3 product. It:

- finds overlapping cycles,
- extracts offshore segments from both products,
- computes native-support and matched-support spectral averages,
- compares the resulting spectra with slope guides.

### Step 6: Run diagnostics on smoothed spectra

Several diagnostic plots help show how spectral smoothness changes with averaging choices, including:

- a reduced-cycle average,
- the set of individual spectra contributing to that average,
- a single-segment example,
- comparison plots between full-ensemble, reduced-ensemble, and single-segment smoothed spectra.

### Step 7: Compare left and right swath halves by cycle

For each unsmoothed cycle, the notebook:

- extracts one offshore 600 km segment from every usable pixel in the left and right swath halves,
- averages spectra separately for the left and right sides,
- compares a few random cycles,
- plots all cycle-mean left/right spectra using a cycle colormap.

## Main Inputs

The notebook depends on these data and local modules:

- unsmoothed SWOT L3 LR SSH Expert files under `/project/downloads/Swot/L3_unsmoothed/swot_l3_expert/`,
- 2 km smoothed SWOT files under `/project/downloads/Swot/l3_karin_nadir/l3_lr_ssh/v3_0/1day_orbit/`,
- local helper modules such as `swot_cycle_reader.py` and `swot_l3_assembler.py`.

It is intended to run in the local Anaconda Python environment used in the notebook, with packages including `numpy`, `pandas`, `xarray`, `netCDF4`, `matplotlib`, and `scipy`.

## Main Outputs

The notebook produces:

- pass-intersection summary tables for cycle 503,
- a map of the selected California pass,
- SSHA summary statistics inside the California bounding box,
- one-cycle and multi-cycle SSH power spectra,
- unsmoothed versus smoothed product comparisons,
- diagnostic plots for ensemble averaging effects,
- per-cycle left/right-swath spectral comparisons.

## Scope Note

This notebook is both exploratory and analytical. It begins as a cycle-503 California pass finder, but it also includes broader spectral analysis across many cycles and a comparison between unsmoothed and smoothed SWOT SSH products.