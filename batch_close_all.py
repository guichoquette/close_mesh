#!/usr/bin/env python3
"""
batch_close_all.py
------------------
Close every epoch of every BraDiPho specimen found in the data folder, and
summarise the results in a CSV file.

Input:  <data>/*_epo-all_dissection/bradipho_data/<specimen>/dissection/*_dissection.ply
Output: <out>/<specimen>/<epoch>_closed.ply  (one closed mesh per epoch)
        <out>/results.csv                    (one line per epoch)

Each epoch goes through `close_mesh_holes`, exactly as with the command line
script. Epochs whose output already exists are skipped unless --overwrite.

Usage:
    python batch_close_all.py [--data brain_data] [--out brain_data/output] [--workers 4]
"""

import argparse
import contextlib
import csv
import io
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CSV_FIELDS = ["specimen", "hemisphere", "epoch", "watertight", "volume", "curved_caps",
              "seconds", "input", "output", "error"]


def find_epochs(data_dir: Path):
    """List the BraDiPho epoch meshes, sorted by specimen and epoch."""
    return sorted(data_dir.glob("*_epo-all_dissection/bradipho_data/*/dissection/*_dissection.ply"))


def describe(path: Path):
    """Specimen, hemisphere and epoch read from a name like spc-01_hem-R_epo-03b_dissection.ply."""
    match = re.match(r"(spc-\d+)_hem-([LR])_epo-(\w+?)_dissection$", path.stem)
    if match is None:
        return path.stem, "", ""
    return match.groups()


def close_one(input_path: str, output_path: str, mesh_filter: float) -> dict:
    """Close one epoch (run in a worker process) and return its CSV row."""
    from close_opening_scripts.close_planar_hole import close_mesh_holes

    specimen, hemisphere, epoch = describe(Path(input_path))
    row = {"specimen": specimen, "hemisphere": hemisphere, "epoch": epoch,
           "input": input_path, "output": output_path, "error": ""}
    args = argparse.Namespace(input=input_path, output=output_path, capsOnly=False, debug=None,
                              debug_loop="", seedStepFactor=4.0, seedDecimator=4.0, meshFilter=mesh_filter)
    start = time.time()
    log = io.StringIO()
    try:
        with contextlib.redirect_stdout(log):
            watertight, volume = close_mesh_holes(args)
        row["watertight"] = watertight
        row["volume"] = f"{volume:.6e}" if volume is not None else ""
    except Exception as error:  # noqa: BLE001 - one failing epoch must not stop the batch
        row["watertight"] = False
        row["volume"] = ""
        row["error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    # The script prints one warning per opening closed with a curved cap
    row["curved_caps"] = log.getvalue().count("closed with a curved cap")
    row["seconds"] = round(time.time() - start)
    return row


def main():
    parser = argparse.ArgumentParser(description="Close every epoch of every BraDiPho specimen.")
    parser.add_argument("--data", default=str(ROOT / "brain_data"), help="Folder containing the *_epo-all_dissection folders.")
    parser.add_argument("--out", default=str(ROOT / "brain_data" / "output"), help="Output folder.")
    parser.add_argument("--workers", type=int, default=4, help="Epochs processed in parallel (default: 4).")
    parser.add_argument("--meshFilter", type=float, default=1e-5, help="Short edge collapse tolerance (default: 1e-5).")
    parser.add_argument("--overwrite", action="store_true", help="Process epochs whose output already exists.")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data).resolve(), Path(args.out).resolve()
    epochs = find_epochs(data_dir)
    if not epochs:
        raise SystemExit(f"No *_dissection.ply file found under {data_dir}")

    # Keep the rows of epochs already done in a previous run
    csv_path = out_dir / "results.csv"
    rows = {}
    if csv_path.exists() and not args.overwrite:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = {r["input"]: r for r in csv.DictReader(f)}

    jobs = []
    for path in epochs:
        specimen, _, _ = describe(path)
        output = out_dir / specimen / f"{path.stem.replace('_dissection', '')}_closed.ply"
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and str(path) in rows and not args.overwrite:
            continue
        jobs.append((str(path), str(output)))

    print(f"{len(epochs)} epochs found, {len(jobs)} to process with {args.workers} workers -> {out_dir}")

    def write_csv():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for row in sorted(rows.values(), key=lambda r: (r["specimen"], r["epoch"])):
                writer.writerow(row)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(close_one, inp, outp, args.meshFilter) for inp, outp in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows[row["input"]] = row
            write_csv()  # written after each epoch, so an interrupted run keeps its results
            status = "error: " + row["error"] if row["error"] else f"watertight={row['watertight']} volume={row['volume'] or 'N/A'}"
            print(f"[{done}/{len(jobs)}] {row['specimen']} epo-{row['epoch']}: {status} ({row['seconds']}s)", flush=True)

    write_csv()
    closed = sum(str(r["watertight"]) == "True" for r in rows.values())
    print(f"Done: {closed}/{len(rows)} epochs watertight. Results: {csv_path}")


if __name__ == "__main__":
    main()
