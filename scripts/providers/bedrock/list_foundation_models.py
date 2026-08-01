#!/usr/bin/env -S uv run --env-file .env --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "botocore>=1.34",
# ]
# ///
"""Scrape the AWS Bedrock foundation model catalog.

Calls the Bedrock control-plane ListFoundationModels API and writes a JSON
snapshot to data/providers/bedrock/foundation_models.json so that git diffs
reflect only real catalog changes (git-scraping technique). The snapshot is a
JSON object keyed by region, each region an object keyed by modelId whose value
is the full model summary:

    {"us-east-1": {"<modelId>": {<summary>}, ...}, ...}

Each run scrapes one region and merges its result into the existing file,
leaving other regions untouched, so multiple regions can be tracked over time.
Region keys are sorted; within a region, models are ordered by
modelLifecycle.startOfLifeTime ascending (modelId as a tiebreaker), so the
newest models sit at the bottom.

The shebang loads a .env file (uv's built-in --env-file) from the current
working directory, so run this from the repo root. Credentials/region are then
resolved by botocore in the usual way (env vars from .env, shared config, instance
role). Region defaults to us-east-1, which carries the widest catalog; override
with AWS_REGION or --region.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import botocore.session

# Shared canonicalizer (lib/common.py). Repo root on sys.path so this
# stdlib-only helper imports without a PEP 723 dependency entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.common import DriftGate, normalize_deep

DEFAULT_REGION = "us-east-1"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "bedrock" / "foundation_models.json"

def fetch_foundation_models(region: str) -> list[dict]:
    """Return every foundation model summary from the Bedrock catalog.

    Uses botocore directly (boto3 is just a wrapper and would drag in
    s3transfer); the client API is identical for our purposes.
    """
    client = botocore.session.get_session().create_client(
        "bedrock", region_name=region
    )
    response = client.list_foundation_models()
    return response.get("modelSummaries", [])


def read_snapshot(path: Path) -> dict[str, dict]:
    """Return the existing region->catalog snapshot, or {} if absent/corrupt.

    A missing or corrupt/partial file is treated as empty so a bad snapshot
    never blocks a fresh scrape; the scraped region is rewritten cleanly and
    other regions are simply re-populated on their next run.
    """
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            snapshot = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return snapshot if isinstance(snapshot, dict) else {}


def vanished_ids(models: list[dict], known: set[str]) -> list[str]:
    """Committed modelIds the API no longer returns (would mutate the file).

    New models are fine — they slot in by startOfLifeTime and appear as pure
    additions. But a model we already represent disappearing would drop it from
    the file, so that fails the run for human review.
    """
    return sorted(known - {m.get("modelId", "") for m in models})


def order_by_lifecycle(models: list[dict]) -> dict[str, dict]:
    """Map of modelId -> summary, ordered oldest-first by startOfLifeTime.

    Newest models end up at the bottom of the file. modelId breaks ties (models
    sharing a startOfLifeTime). Any model missing startOfLifeTime sorts after
    the dated ones, ordered by modelId, so ordering never crashes on absence.
    """
    def sort_key(m: dict):
        lifecycle = m.get("modelLifecycle") or {}
        start = lifecycle.get("startOfLifeTime")
        return (start is None, start, m.get("modelId", ""))

    return {m.get("modelId", ""): m for m in sorted(models, key=sort_key)}


# Leading keys for each model summary, in the order they should appear; any
# remaining keys follow alphabetically.
MODEL_KEY_ORDER = (
    "modelName",
    "providerName",
    "modelId",
    "modelArn",
    "modelLifecycle",
)


def order_model_keys(model: dict) -> dict:
    """Order a model summary's top-level keys (preferred first, then A-Z).

    Nested values are deep-sorted so diffs below the top level stay stable.
    """
    rest = sorted(k for k in model if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in model] + rest
    return {k: normalize_deep(model[k]) for k in keys}


def write_csv(snapshot: dict[str, dict], path: Path) -> None:
    """Render the snapshot as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["region", "modelId", "modelName", "providerName", "startOfLifeTime", "status"])
    for region, catalog in snapshot.items():
        for mid, m in catalog.items():
            lifecycle = m.get("modelLifecycle") or {}
            writer.writerow([
                region,
                mid,
                m.get("modelName", ""),
                m.get("providerName", ""),
                str(lifecycle.get("startOfLifeTime", ""))[:10],
                lifecycle.get("status", ""),
            ])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", DEFAULT_REGION),
        help=f"AWS region to query (default: %(default)s)",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Accept any drift instead of erroring (skips the marker handshake)",
    )
    parser.add_argument(
        "--accept-pending",
        action="store_true",
        help="Accept exactly the drift recorded in the pending-drift marker "
        "(the workflow passes this on re-runs — the ack half of the handshake)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the JSON snapshot (default: %(default)s)",
    )
    args = parser.parse_args()

    models = fetch_foundation_models(args.region)
    snapshot = read_snapshot(args.output)

    # Drift detection: the committed catalog is the source of truth for what we
    # already represent in this region. New models are free (pure additions),
    # but a model we already have vanishing upstream would drop it from the
    # file, so that fails for human review, acknowledged through the DriftGate
    # handshake. The kind is region-qualified so an acknowledgement recorded
    # for one region can never authorize a removal in another.
    gate = DriftGate(args.output, allow_all=args.allow_drift, accept_pending=args.accept_pending)
    drift = gate.unacked(
        f"vanished:{args.region}", vanished_ids(models, set(snapshot.get(args.region, {})))
    )
    if drift:
        print(
            f"error: {len(drift)} committed model(s) no longer returned by the API:",
            file=sys.stderr,
        )
        for mid in drift:
            print(f"  {mid}", file=sys.stderr)
        gate.record()
        return 1

    catalog = order_by_lifecycle(models)
    catalog = {mid: order_model_keys(model) for mid, model in catalog.items()}

    # Merge this region into the existing snapshot, then sort region keys so the
    # top level has a stable order regardless of scrape sequence.
    snapshot[args.region] = catalog
    snapshot = {region: snapshot[region] for region in sorted(snapshot)}

    # Region and modelId ordering is intentional, so no sort_keys; each model's
    # inner keys are already sorted by sort_keys_deep. botocore returns datetime
    # objects (e.g. modelLifecycle.startOfLifeTime) that json can't serialize,
    # so default=str renders them as ISO-8601 strings.
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False, default=str)

    # Write atomically so a serialization error or interrupt can never leave a
    # truncated snapshot behind for the next run to choke on.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(args.output)

    write_csv(snapshot, args.output.with_suffix(".csv"))
    gate.clear()

    print(
        f"Wrote {len(catalog)} foundation models for {args.region} to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
