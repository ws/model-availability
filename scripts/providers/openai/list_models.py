#!/usr/bin/env -S uv run --env-file .env --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai>=2.45",
# ]
# ///
"""Scrape the OpenAI first-party model catalog.

Queries the Models API (GET /v1/models) via the openai SDK. Auth is keyless
only: OpenAI Workload Identity Federation. OPENAI_IDENTITY_PROVIDER_ID and
OPENAI_SERVICE_ACCOUNT_ID must be set, and the GitHub Actions OIDC endpoint
(ACTIONS_ID_TOKEN_REQUEST_URL/_TOKEN) must be available — the GitHub-minted
OIDC token is exchanged for a short-lived OpenAI access token. This means the
scraper only runs inside GitHub Actions (or anywhere else that can mint an OIDC
token matching the Workload Identity Provider's configuration).

The snapshot is written to data/providers/openai/models.json as a JSON object
keyed by model id (the API is global — no region dimension), ordered by
`created` ascending (id as a tiebreaker) so the newest models sit at the
bottom. A sibling models.csv is written for human review.

The `created` timestamp is pinned when it changes by less than 24 hours;
larger changes require acknowledgement through the DriftGate handshake.

The shebang loads a .env file (uv's built-in --env-file) from the current
working directory, so run this from the repo root.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import sys
from pathlib import Path

import urllib.parse

import httpx  # dependency of the openai SDK
import openai
from openai.auth import SubjectTokenProvider

# Shared canonicalizer (lib/common.py). Repo root on sys.path so this
# stdlib-only helper imports without a PEP 723 dependency entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.common import DriftGate, normalize_deep

DEFAULT_WIF_AUDIENCE = "https://api.openai.com/v1"

# `created` is an immutable creation timestamp. For a model we already
# represent, keep the committed value when the API's value is within this window
# (absorbs any harmless wobble and keeps the `created` ordering stable); a jump
# beyond it fails the run for review, since it shouldn't happen.
CREATED_DRIFT_SECONDS = 24 * 3600

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "openai" / "models.json"


def github_actions_oidc_token_provider(audience: str) -> SubjectTokenProvider:
    """SubjectTokenProvider that mints the GitHub Actions OIDC token.

    The SDK ships k8s/azure/gcp providers but not a GitHub Actions one; the
    interface is just {token_type, get_token}. GitHub's token is an OIDC id
    token, same as GCP's ("id").
    """
    def get_token() -> str:
        request_url = os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"]
        request_token = os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]
        url = f"{request_url}&audience={urllib.parse.quote(audience, safe='')}"
        response = httpx.get(
            url, headers={"Authorization": f"bearer {request_token}"}, timeout=30
        )
        response.raise_for_status()
        return response.json()["value"]

    return {"token_type": "id", "get_token": get_token}


def build_client() -> openai.OpenAI:
    """Return an OpenAI client authenticated via Workload Identity Federation."""
    if not (
        os.environ.get("OPENAI_IDENTITY_PROVIDER_ID")
        and os.environ.get("OPENAI_SERVICE_ACCOUNT_ID")
        and os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    ):
        raise RuntimeError(
            "WIF not available: requires OPENAI_IDENTITY_PROVIDER_ID, "
            "OPENAI_SERVICE_ACCOUNT_ID, and the GitHub Actions OIDC endpoint "
            "(ACTIONS_ID_TOKEN_REQUEST_URL) — run inside GitHub Actions."
        )
    return openai.OpenAI(
        workload_identity={
            "identity_provider_id": os.environ["OPENAI_IDENTITY_PROVIDER_ID"],
            "service_account_id": os.environ["OPENAI_SERVICE_ACCOUNT_ID"],
            "provider": github_actions_oidc_token_provider(
                os.environ.get("OPENAI_WIF_AUDIENCE", DEFAULT_WIF_AUDIENCE)
            ),
        },
    )


def fetch_models() -> list[dict]:
    """Return every model from the Models API (the SDK auto-paginates)."""
    client = build_client()
    return [json.loads(m.to_json()) for m in client.models.list()]


def reconcile_created(models: list[dict], committed: dict[str, int]) -> list[str]:
    """Pin `created` to the committed value; return ids that jumped > 24h.

    `created` should be immutable, so for a model we already represent we keep
    the committed value when the API's is within CREATED_DRIFT_SECONDS — that
    absorbs any harmless wobble and keeps ordering stable. A larger move is
    collected so the caller can fail for human review. Mutates `models`.
    """
    drifted: list[str] = []
    for m in models:
        old = committed.get(m.get("id", ""))
        new = m.get("created")
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            continue  # new model, or non-numeric — nothing to pin against
        if abs(new - old) > CREATED_DRIFT_SECONDS:
            drifted.append(m.get("id", ""))
        else:
            m["created"] = old  # pin to the committed value
    return sorted(drifted)


def order_by_created(models: list[dict]) -> dict[str, dict]:
    """Map of id -> model, ordered oldest-first by `created` (id ties)."""
    def sort_key(m: dict):
        created = m.get("created")
        return (created is None, created, m.get("id", ""))

    return {m.get("id", ""): m for m in sorted(models, key=sort_key)}


# Leading keys for each model, in the order they should appear; any remaining
# keys follow alphabetically.
MODEL_KEY_ORDER = ("id", "created")

# Keys dropped from the output. `object` is the constant "model" for every row,
# and `owned_by` is a coarse org label ("openai"/"system"/...) that adds nothing
# to a catalog keyed by id.
DROP_KEYS = {"object", "owned_by"}


def order_model_keys(model: dict) -> dict:
    """Order a model's top-level keys (preferred first, then A-Z), deep-sorted.

    Keys in DROP_KEYS are omitted from the output.
    """
    present = [k for k in model if k not in DROP_KEYS]
    rest = sorted(k for k in present if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in present] + rest
    return {k: normalize_deep(model[k]) for k in keys}


def read_existing_created(path: Path) -> dict[str, int]:
    """Map of committed model id -> `created` ({} if the file is absent)."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return {mid: m.get("created") for mid, m in json.load(f).items()}
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}


def write_csv(catalog: dict[str, dict], path: Path) -> None:
    """Render the catalog as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["id", "created"])
    for mid, m in catalog.items():
        created = m.get("created")
        date = (
            datetime.datetime.fromtimestamp(created, datetime.UTC).date()
            if isinstance(created, (int, float))
            else ""
        )
        writer.writerow([mid, date])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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

    try:
        models = fetch_models()
    except openai.APIStatusError as err:
        print(f"HTTP {err.status_code}\n{err.message}", file=sys.stderr)
        return 1
    except (openai.OpenAIError, RuntimeError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1

    # `created` is immutable, so a material timestamp change requires review.
    gate = DriftGate(args.output, allow_all=args.allow_drift, accept_pending=args.accept_pending)
    created_drift = gate.unacked(
        "created", reconcile_created(models, read_existing_created(args.output))
    )
    if created_drift:
        print(
            f"error: {len(created_drift)} committed model(s) changed "
            "`created` by more than 24h — it should be immutable:",
            file=sys.stderr,
        )
        for mid in created_drift:
            print(f"  {mid}", file=sys.stderr)
        gate.record()
        return 1

    catalog = order_by_created(models)
    catalog = {mid: order_model_keys(m) for mid, m in catalog.items()}

    # Id ordering is intentional, so no sort_keys; inner keys are already
    # normalized by order_model_keys/normalize_deep.
    payload = json.dumps(catalog, indent=2, ensure_ascii=False)

    # Write atomically so a failure can never leave a truncated snapshot behind.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(args.output)

    write_csv(catalog, args.output.with_suffix(".csv"))
    gate.clear()

    print(f"Wrote {len(catalog)} OpenAI models to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
