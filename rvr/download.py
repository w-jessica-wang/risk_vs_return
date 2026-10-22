# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Mirror Binance's monthly USDⓈ-M funding-rate and 1h kline archives into data/raw.

Each archive is written only after its SHA-256 matches Binance's .CHECKSUM sidecar, then
renamed into place atomically, so any archive on disk is a verified archive and re-running
the command resumes where it stopped. A listing or download that fails is logged to
data/raw/_failures.tsv and retried on the next run; the exit code is non-zero until none remain.

Run:  python -m rvr.download --dry-run            # list archives and total size only
      python -m rvr.download --symbols BTCUSDT    # smoke test
      python -m rvr.download                      # every USDT-margined perp, live or delisted
"""
import argparse
import hashlib
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rvr import listing, net
from rvr.config import (DATA_BASE_URL, DATASET_PREFIXES, QUOTE_ASSET, RAW_DIR, SAMPLE_END,
                        SAMPLE_START, UM_MONTHLY_PREFIX)


class ChecksumError(ValueError):
    """An archive's SHA-256 did not match Binance's published checksum."""


def verify_checksum(payload, checksum_text):
    fields = checksum_text.split()
    return bool(fields) and hashlib.sha256(payload).hexdigest() == fields[0].lower()


def local_path(key, raw_dir=RAW_DIR):
    prefix = f"{UM_MONTHLY_PREFIX}/"
    if not key.startswith(prefix) or ".." in key:
        raise ValueError(f"unexpected archive key: {key}")
    return Path(raw_dir) / key[len(prefix):]


def fetch_file(key, raw_dir=RAW_DIR, get_bytes=net.get_bytes, get_text=net.get_text):
    """Download one archive into raw_dir. Returns "cached" or "downloaded"."""
    dest = local_path(key, raw_dir)
    if dest.exists():
        return "cached"
    url = f"{DATA_BASE_URL}/{key}"
    payload = get_bytes(url)
    if not verify_checksum(payload, get_text(url + ".CHECKSUM")):
        raise ChecksumError(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(payload)
    partial.replace(dest)
    return "downloaded"


def _run_all(jobs, work, workers, label):
    """Run work(job) over jobs in threads → (results in job order, [(job, error)]), with progress."""
    results, failures = [None] * len(jobs), []
    step = max(1, len(jobs) // 20)
    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(work, job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), 1):
            i = futures[future]
            try:
                results[i] = future.result()
            except Exception as exc:  # keep going: every failure is logged and retried on the next run
                failures.append((jobs[i], repr(exc)))
            if done % step == 0 or done == len(jobs):
                print(f"  {label} {done:,}/{len(jobs):,}  failed={len(failures)}  "
                      f"{time.monotonic() - started:,.0f}s", flush=True)
    return results, failures


def plan(symbols, datasets, start, end, workers, fetch=None):
    """Every monthly archive to mirror → ([(dataset, key, size)], [(listing job, error)])."""
    def list_job(job):
        dataset, symbol = job
        return [(dataset, key, size)
                for key, size in listing.list_monthly_files(dataset, symbol, start, end, fetch=fetch)]

    jobs = [(dataset, symbol) for symbol in symbols for dataset in datasets]
    results, failures = _run_all(jobs, list_job, workers, "listed")
    files = sorted(item for found in results if found for item in found)
    return files, [(f"listing {dataset} {symbol}", error) for (dataset, symbol), error in failures]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", help="default: every USDT-margined perpetual in the archive")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_PREFIXES), default=sorted(DATASET_PREFIXES))
    parser.add_argument("--start", default=SAMPLE_START, help="first month, YYYY-MM")
    parser.add_argument("--end", default=SAMPLE_END, help="last month, YYYY-MM")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="list archives and total size; download nothing")
    args = parser.parse_args(argv)

    symbols = args.symbols or listing.list_perp_symbols(QUOTE_ASSET)
    print(f"{len(symbols)} symbols, months {args.start} → {args.end}; listing archives…", flush=True)
    files, failures = plan(symbols, args.datasets, args.start, args.end, args.workers)
    totals = defaultdict(lambda: [0, 0])
    for dataset, _, size in files:
        totals[dataset][0] += 1
        totals[dataset][1] += size
    for dataset, (n, size) in sorted(totals.items()):
        print(f"  {dataset:<10} {n:>7,} archives  {size / 1e6:>9,.1f} MB")

    if not args.dry_run:
        keys = [key for _, key, _ in files]
        outcomes, download_failures = _run_all(keys, fetch_file, args.workers, "fetched")
        print(f"  {dict(Counter(o for o in outcomes if o))}", flush=True)
        failures += download_failures

    failure_log = RAW_DIR / "_failures.tsv"
    if failures:
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        failure_log.write_text("".join(f"{job}\t{error}\n" for job, error in sorted(failures)))
        print(f"{len(failures)} listings/archives failed; see {failure_log}. Re-run to retry.", file=sys.stderr)
        return 1
    failure_log.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
