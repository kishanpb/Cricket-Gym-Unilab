"""Retain native TensorBoard scalars as portable CSV and a compact summary."""

import argparse
import csv
import json
import math
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def retain(run_dir: Path) -> None:
    events = EventAccumulator(str(run_dir), size_guidance={"scalars": 0}).Reload()
    tags = sorted(tag for tag in events.Tags()["scalars"] if not tag.endswith("/time"))
    values = {tag: {event.step: event.value for event in events.Scalars(tag)} for tag in tags}
    steps = sorted({step for series in values.values() for step in series})
    with (run_dir / "training_scalars.csv").open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["iteration", *tags])
        writer.writerows([step, *(values[tag].get(step, "") for tag in tags)] for step in steps)
    report = {
        "scope": "native iteration-indexed scalars; redundant wall-time-indexed /time series omitted; no reconstructed KL or clip-fraction estimates",
        "kl_series_available": any("kl" in tag.lower() for tag in tags),
        "clip_fraction_series_available": any("clip_fraction" in tag.lower() for tag in tags),
        "scalar_tags": {
            tag: {
                "count": len(series),
                "all_finite": all(math.isfinite(value) for value in series.values()),
                "minimum": min(series.values()),
                "maximum": max(series.values()),
                "final": series[max(series)],
            }
            for tag, series in values.items()
        },
    }
    (run_dir / "training_diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    retain(parser.parse_args().run_dir)
