"""
SWOT L3 LR SSH Unsmoothed Downloader
====================================
Downloads NetCDF files from the AVISO THREDDS Data Server.

Product: SWOT_L3_LR_SSH_Unsmoothed  (v2.0.1)
Source : https://tds-odatis.aviso.altimetry.fr/thredds/catalog/
         dataset-l3-swot-karin-nadir-validated/l3_lr_ssh/v2_0_1/Unsmoothed/

Authentication
--------------
Register at https://www.aviso.altimetry.fr to obtain credentials, then either:
  • Set environment variables  AVISO_USER  and  AVISO_PASS, or
  • Pass --user / --password on the command line.

Usage
-----
  python download_swot_l3.py --output-dir ./swot_l3 --workers 4
  python download_swot_l3.py --cycles 1 2 3 --output-dir ./swot_l3
  python download_swot_l3.py --dry-run          # list files, don't download

Dependencies
------------
  pip install requests tqdm beautifulsoup4 lxml
"""

import argparse
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = (
    "https://tds-odatis.aviso.altimetry.fr/thredds/"
)
CATALOG_ROOT = (
    "catalog/dataset-l3-swot-karin-nadir-validated/"
    "l3_lr_ssh/v2_0_1/Unsmoothed/"
)
FILESERVER_ROOT = (
    "fileServer/dataset-l3-swot-karin-nadir-validated/"
    "l3_lr_ssh/v2_0_1/Unsmoothed/"
)

# Cycles 474-578 are the 1-day repeat fast-sampling (CalVal) phase
# (orbit manoeuvres to the 21-day science orbit completed July 21 2023).
# Cycles 001-032 are the early 21-day nominal science orbit cycles.
ONE_DAY_CYCLES = list(range(474, 579))
FILENAME_PATTERN = re.compile(r"^SWOT_L3_LR_SSH_Unsmoothed_(\d{3})_(\d{3})_.*\.nc$")

CHUNK_SIZE = 1024 * 1024  # 1 MB
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0  # seconds, exponential
PROGRESS_REPORT_INTERVAL = 10.0  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog crawling
# ---------------------------------------------------------------------------

def catalog_url(cycle: int) -> str:
    """HTML catalog URL for a given cycle number."""
    return urljoin(
        BASE_URL,
        f"{CATALOG_ROOT}cycle_{cycle:03d}/catalog.html",
    )


def file_download_url(cycle: int, filename: str) -> str:
    """Direct-download URL via the THREDDS fileServer."""
    return urljoin(
        BASE_URL,
        f"{FILESERVER_ROOT}cycle_{cycle:03d}/{filename}",
    )


def list_cycle_files(
    session: requests.Session,
    cycle: int,
    tracks: set[int] | None = None,
) -> list[tuple[int, str]]:
    """
    Parse the THREDDS HTML catalog for *cycle* and return
    a list of (cycle, filename) tuples.
    """
    url = catalog_url(cycle)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                log.error("Failed to fetch catalog for cycle %03d: %s", cycle, exc)
                return []
            wait = RETRY_BACKOFF ** attempt
            log.warning(
                "Catalog fetch error (attempt %d/%d), retrying in %.1fs: %s",
                attempt, MAX_RETRIES, wait, exc,
            )
            time.sleep(wait)

    soup = BeautifulSoup(r.text, "lxml")
    files = []
    for tag in soup.find_all("a"):
        name = tag.get_text(strip=True)
        m = FILENAME_PATTERN.match(name)
        if not m:
            continue

        file_track = int(m.group(2))
        if tracks is not None and file_track not in tracks:
            continue

        files.append((cycle, name))

    if tracks is None:
        log.info("Cycle %03d → %d files found.", cycle, len(files))
    else:
        track_list = ",".join(f"{t:03d}" for t in sorted(tracks))
        log.info(
            "Cycle %03d → %d files found (track filter: %s).",
            cycle,
            len(files),
            track_list,
        )
    return files


def collect_all_files(
    session: requests.Session,
    cycles: list[int],
    tracks: set[int] | None = None,
) -> list[tuple[int, str]]:
    """Crawl catalogs for all requested cycles."""
    all_files: list[tuple[int, str]] = []
    for cycle in cycles:
        all_files.extend(list_cycle_files(session, cycle, tracks=tracks))
    log.info("Total files to consider: %d", len(all_files))
    return all_files


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def _already_complete(path: Path, expected_size: int | None) -> bool:
    """Return True if the local file exists and matches the expected size."""
    if not path.exists():
        return False
    if expected_size is not None and path.stat().st_size != expected_size:
        return False
    return True


def _fmt_bytes(nbytes: int | float) -> str:
    """Human-readable byte count."""
    n = float(nbytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    unit_idx = 0
    while n >= 1024.0 and unit_idx < len(units) - 1:
        n /= 1024.0
        unit_idx += 1
    return f"{n:.2f} {units[unit_idx]}"


def download_file(
    session: requests.Session,
    cycle: int,
    filename: str,
    output_dir: Path,
    dry_run: bool = False,
    progress_interval: float = PROGRESS_REPORT_INTERVAL,
) -> tuple[str, str]:
    """
    Download a single NetCDF file with resume support and retries.

    Returns (filename, status) where status is one of:
      'downloaded', 'skipped', 'dry-run', 'failed'
    """
    url = file_download_url(cycle, filename)
    dest_dir = output_dir / f"cycle_{cycle:03d}"
    dest_path = dest_dir / filename

    if dry_run:
        return filename, "dry-run"

    dest_dir.mkdir(parents=True, exist_ok=True)

    # ---- Attempt with retries ----
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # Support resumable downloads via Range header
            existing_bytes = dest_path.stat().st_size if dest_path.exists() else 0
            headers = {}
            if existing_bytes > 0:
                headers["Range"] = f"bytes={existing_bytes}-"

            r = session.get(url, headers=headers, stream=True, timeout=60)

            # 416 = range not satisfiable → file already complete
            if r.status_code == 416:
                return filename, "skipped"

            r.raise_for_status()

            # Check Content-Length to skip complete files
            total = int(r.headers.get("Content-Length", 0)) or None
            if total is not None and existing_bytes > 0:
                expected_total = existing_bytes + total
            else:
                expected_total = total

            if existing_bytes > 0 and total is not None and total == 0:
                return filename, "skipped"

            mode = "ab" if existing_bytes > 0 else "wb"
            start_time = time.monotonic()
            last_report = start_time
            bytes_since_report = 0
            bytes_written = 0

            with open(dest_path, mode) as fh:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
                        chunk_len = len(chunk)
                        bytes_written += chunk_len
                        bytes_since_report += chunk_len

                        now = time.monotonic()
                        if now - last_report >= progress_interval:
                            elapsed = max(now - start_time, 1e-9)
                            report_elapsed = max(now - last_report, 1e-9)
                            transferred = existing_bytes + bytes_written
                            avg_rate_mbps = (bytes_written / (1024.0 * 1024.0)) / elapsed
                            inst_rate_mbps = (bytes_since_report / (1024.0 * 1024.0)) / report_elapsed

                            if expected_total:
                                pct = 100.0 * transferred / expected_total
                                log.info(
                                    "%s: %.1f%% (%s/%s), avg %.2f MB/s, now %.2f MB/s",
                                    filename,
                                    pct,
                                    _fmt_bytes(transferred),
                                    _fmt_bytes(expected_total),
                                    avg_rate_mbps,
                                    inst_rate_mbps,
                                )
                            else:
                                log.info(
                                    "%s: transferred %s, avg %.2f MB/s, now %.2f MB/s",
                                    filename,
                                    _fmt_bytes(transferred),
                                    avg_rate_mbps,
                                    inst_rate_mbps,
                                )

                            last_report = now
                            bytes_since_report = 0

            elapsed_total = max(time.monotonic() - start_time, 1e-9)
            avg_rate_mbps = (bytes_written / (1024.0 * 1024.0)) / elapsed_total
            final_size = existing_bytes + bytes_written
            if expected_total:
                pct = 100.0 * final_size / expected_total
                log.info(
                    "%s: complete %.1f%% (%s/%s) in %.1fs, avg %.2f MB/s",
                    filename,
                    pct,
                    _fmt_bytes(final_size),
                    _fmt_bytes(expected_total),
                    elapsed_total,
                    avg_rate_mbps,
                )
            else:
                log.info(
                    "%s: complete %s in %.1fs, avg %.2f MB/s",
                    filename,
                    _fmt_bytes(final_size),
                    elapsed_total,
                    avg_rate_mbps,
                )

            return filename, "downloaded"

        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                log.error("FAILED %s: %s", filename, exc)
                return filename, "failed"
            wait = RETRY_BACKOFF ** attempt
            log.warning(
                "%s – error on attempt %d/%d, retrying in %.1fs: %s",
                filename, attempt, MAX_RETRIES, wait, exc,
            )
            time.sleep(wait)

    return filename, "failed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download SWOT L3 LR SSH Unsmoothed files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output-dir", "-o",
        default="./swot_l3_expert",
        help="Root directory for downloaded files.",
    )
    p.add_argument(
        "--cycles", "-c",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Cycle numbers to download. Example: --cycles 474 475 480. "
            "If omitted, uses default one-day cycles and optional cycle bounds."
        ),
    )
    p.add_argument(
        "--cycle-min",
        type=int,
        default=None,
        help="Minimum cycle to include (e.g. --cycle-min 475).",
    )
    p.add_argument(
        "--cycle-max",
        type=int,
        default=None,
        help="Maximum cycle to include.",
    )
    p.add_argument(
        "--tracks", "-t",
        type=int,
        nargs="+",
        default=None,
        help="Track numbers to include (from filename), e.g. --tracks 13 or --tracks 13 26.",
    )
    p.add_argument(
        "--workers", "-w",
        type=int, default=4,
        help="Number of parallel download threads.",
    )
    p.add_argument(
        "--user", "-u",
        default=os.environ.get("AVISO_USER", ""),
        help="AVISO username (or set env var AVISO_USER).",
    )
    p.add_argument(
        "--password", "-p",
        default=os.environ.get("AVISO_PASS", ""),
        help="AVISO password (or set env var AVISO_PASS).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without actually fetching them.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--progress-interval",
        type=float,
        default=PROGRESS_REPORT_INTERVAL,
        help="Seconds between per-file transfer progress reports.",
    )
    return p.parse_args()


def build_session(user: str, password: str) -> requests.Session:
    session = requests.Session()
    if user and password:
        session.auth = (user, password)
    else:
        log.warning(
            "No credentials provided. Set AVISO_USER / AVISO_PASS or "
            "use --user / --password. Downloads may fail if auth is required."
        )
    session.headers.update({"User-Agent": "swot-l3-downloader/1.0"})
    return session


def resolve_cycles(args: argparse.Namespace) -> list[int]:
    """Resolve cycle selection from explicit list and/or min/max bounds."""
    cycles = sorted(set(args.cycles)) if args.cycles else ONE_DAY_CYCLES.copy()

    if args.cycle_min is not None:
        cycles = [c for c in cycles if c >= args.cycle_min]
    if args.cycle_max is not None:
        cycles = [c for c in cycles if c <= args.cycle_max]

    if args.cycle_min is not None and args.cycle_max is not None and args.cycle_min > args.cycle_max:
        raise ValueError("--cycle-min cannot be greater than --cycle-max")
    if not cycles:
        raise ValueError("No cycles selected after applying --cycles/--cycle-min/--cycle-max")

    return cycles


def main() -> None:
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    try:
        selected_cycles = resolve_cycles(args)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(2)

    selected_tracks = set(args.tracks) if args.tracks else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = build_session(args.user, args.password)

    # ---- Crawl catalogs ----
    log.info(
        "Crawling THREDDS catalogs for cycles: %s",
        ", ".join(f"{c:03d}" for c in selected_cycles),
    )
    if selected_tracks is not None:
        log.info(
            "Applying track filter: %s",
            ", ".join(f"{t:03d}" for t in sorted(selected_tracks)),
        )

    all_files = collect_all_files(session, selected_cycles, tracks=selected_tracks)

    if not all_files:
        log.error("No files found. Check credentials and cycle numbers.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {len(all_files)} files would be downloaded:")
        print(f"{'='*60}")
        for cycle, fname in all_files:
            print(f"  cycle_{cycle:03d}/{fname}")
        return

    # ---- Download in parallel ----
    log.info(
        "Downloading %d files → %s  (workers=%d)",
        len(all_files), output_dir, args.workers,
    )

    stats = {"downloaded": 0, "skipped": 0, "failed": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_file,
                session,
                cycle,
                fname,
                output_dir,
                False,
                args.progress_interval,
            ): (cycle, fname)
            for cycle, fname in all_files
        }

        with tqdm(total=len(futures), unit="file", desc="Downloading") as pbar:
            for future in as_completed(futures):
                fname, status = future.result()
                stats[status] = stats.get(status, 0) + 1
                pbar.set_postfix(
                    dl=stats["downloaded"],
                    skip=stats["skipped"],
                    fail=stats["failed"],
                )
                pbar.update(1)
                if status == "failed":
                    log.warning("FAILED: %s", fname)

    log.info(
        "Done. downloaded=%d  skipped=%d  failed=%d",
        stats["downloaded"], stats["skipped"], stats["failed"],
    )
    if stats["failed"] > 0:
        log.warning(
            "%d files failed. Re-run the script to retry (resume is supported).",
            stats["failed"],
        )


if __name__ == "__main__":
    main()
