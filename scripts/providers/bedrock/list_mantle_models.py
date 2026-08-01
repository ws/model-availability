#!/usr/bin/env -S uv run --env-file .env --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "botocore>=1.34",
# ]
# ///
"""Scrape the AWS Bedrock "mantle" model catalog.

There is no SDK client for this endpoint, so we issue a SigV4-signed GET to
    https://bedrock-mantle.<region>.api.aws/v1/models
The API is OpenAI-compatible; the project is selected via the OpenAI-Project
header (see the Bedrock projects docs) and the response is
{"object": "list", "data": [{"id": ..., "created": <unix>, ...}, ...]}.

The snapshot is written to data/providers/bedrock/mantle_models.json as a JSON
object keyed by region, each region an object keyed by model id:

    {"us-east-1": {"<id>": {<model>}, ...}, ...}

Each run scrapes one region and merges its result into the existing file,
leaving other regions untouched. Region keys are sorted; within a region,
models are ordered by `created` ascending (id as a tiebreaker) so the newest
models sit at the bottom.

The shebang loads a .env file (uv's built-in --env-file) from the current
working directory, so run this from the repo root. Credentials/region are
resolved by botocore in the usual way (env vars from .env, shared config, instance
role). Region defaults to us-east-1; override with AWS_REGION or --region.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import botocore.session
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest

# Shared canonicalizer (lib/common.py). Repo root on sys.path so this
# stdlib-only helper imports without a PEP 723 dependency entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.common import DriftGate, normalize_deep

DEFAULT_REGION = "us-east-1"

# SigV4 signing name for the Bedrock control plane.
SIGNING_SERVICE = "bedrock"

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "bedrock" / "mantle_models.json"


def fetch_mantle_models(region: str, project: str) -> dict:
    """Return the raw JSON from the mantle /v1/models endpoint.

    The API is OpenAI-compatible; the project is selected via the OpenAI-Project
    header (a request without it resolves to the account's "default" project).
    """
    url = f"https://bedrock-mantle.{region}.api.aws/v1/models"

    credentials = botocore.session.get_session().get_credentials()
    if credentials is None:
        raise RuntimeError("No AWS credentials found (check .env / shared config).")

    request = AWSRequest(method="GET", url=url, headers={"OpenAI-Project": project})
    SigV4Auth(credentials, SIGNING_SERVICE, region).add_auth(request)
    prepared = request.prepare()

    signed = urllib.request.Request(
        prepared.url, headers=dict(prepared.headers), method="GET"
    )
    with urllib.request.urlopen(signed) as response:
        return json.load(response)


def read_snapshot(path: Path) -> dict[str, dict]:
    """Return the existing region->catalog snapshot, or {} if absent/corrupt."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            snapshot = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return snapshot if isinstance(snapshot, dict) else {}


def vanished_ids(models: list[dict], known: set[str]) -> list[str]:
    """Committed ids the API no longer returns (would mutate the file).

    New models are fine — they slot in by `created` and appear as pure
    additions. But a model we already represent disappearing would drop it from
    the file, so that fails the run for human review.
    """
    return sorted(known - {m.get("id", "") for m in models})


def order_by_created(models: list[dict]) -> dict[str, dict]:
    """Map of id -> model, ordered oldest-first by `created` (id breaks ties).

    Newest models end up at the bottom of the file. A model missing `created`
    sorts after the dated ones, ordered by id, so ordering never crashes.
    """
    def sort_key(m: dict):
        created = m.get("created")
        return (created is None, created, m.get("id", ""))

    return {m.get("id", ""): m for m in sorted(models, key=sort_key)}


# Leading keys for each model, in the order they should appear; any remaining
# keys follow alphabetically.
MODEL_KEY_ORDER = ("id", "created", "owned_by", "status", "status_reason")


def order_model_keys(model: dict) -> dict:
    """Order a model's top-level keys (preferred first, then A-Z), deep-sorted."""
    rest = sorted(k for k in model if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in model] + rest
    return {k: normalize_deep(model[k]) for k in keys}


def write_csv(snapshot: dict[str, dict], path: Path) -> None:
    """Render the snapshot as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["region", "id", "created", "status", "owned_by"])
    for region, catalog in snapshot.items():
        for mid, m in catalog.items():
            created = m.get("created")
            date = (
                datetime.datetime.fromtimestamp(created, datetime.UTC).date()
                if isinstance(created, (int, float))
                else ""
            )
            writer.writerow([region, mid, date, m.get("status", ""), m.get("owned_by", "")])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", DEFAULT_REGION),
        help="AWS region to query (default: %(default)s)",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("AWS_BEDROCK_MANTLE_PROJECT_ID"),
        help="Bedrock mantle project id (default: $AWS_BEDROCK_MANTLE_PROJECT_ID)",
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

    if not args.project:
        print(
            "No project id: pass --project or set AWS_BEDROCK_MANTLE_PROJECT_ID.",
            file=sys.stderr,
        )
        return 1

    try:
        response = fetch_mantle_models(args.region, args.project)
    except urllib.error.HTTPError as err:
        # Surface the response body — invaluable while we're still learning the
        # endpoint (wrong signing service, region, path, etc.).
        print(f"HTTP {err.code} {err.reason}\n{err.read().decode()}", file=sys.stderr)
        return 1

    models = response.get("data", [])
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

    catalog = order_by_created(models)
    catalog = {mid: order_model_keys(model) for mid, model in catalog.items()}

    # Merge this region into the existing snapshot, then sort region keys so the
    # top level has a stable order regardless of scrape sequence.
    snapshot[args.region] = catalog
    snapshot = {region: snapshot[region] for region in sorted(snapshot)}

    # Region and id ordering is intentional, so no sort_keys; inner keys are
    # already normalized by order_model_keys/normalize_deep.
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False)

    # Write atomically so a failure can never leave a truncated snapshot behind.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(args.output)

    write_csv(snapshot, args.output.with_suffix(".csv"))
    gate.clear()

    print(
        f"Wrote {len(catalog)} mantle models for {args.region} to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
