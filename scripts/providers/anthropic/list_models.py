#!/usr/bin/env -S uv run --env-file .env --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "anthropic>=0.90",
# ]
# ///
"""Scrape the Anthropic first-party model catalog.

Queries the Models API (GET /v1/models) via the anthropic SDK. Auth is keyless:
in CI, Workload Identity Federation with the GitHub Actions OIDC token (the
ANTHROPIC_FEDERATION_RULE_ID / ANTHROPIC_ORGANIZATION_ID /
ANTHROPIC_SERVICE_ACCOUNT_ID / ANTHROPIC_IDENTITY_TOKEN[_FILE] env vars — a rule
scoped to workspace:inference covers the Models API); locally, an
`ant auth login` OAuth profile. Both resolve automatically via the SDK's
zero-arg credential chain.

The snapshot is written to data/providers/anthropic/models.json as a JSON
object keyed by model id (the API is global — no region dimension), ordered by
created_at ascending (id as a tiebreaker) so the newest models sit at the
bottom. Each model's `capabilities` is collapsed from the API's nested
{"supported": bool} tree to a flat sorted list of supported paths (e.g.
"effort/xhigh", "thinking/types/adaptive"). A sibling models.csv is written for
human review.

Drift detection: the committed catalog is the source of truth for what we
already represent. New models are free (pure additions), but a model we already
have vanishing upstream fails the run for human review. Acknowledge via the
DriftGate handshake (re-run the failed workflow, or --accept-pending locally)
to accept exactly the recorded drift, or --allow-drift to accept everything.

The shebang loads a .env file (uv's built-in --env-file) from the current
working directory, so run this from the repo root.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import sys
from pathlib import Path

import anthropic

# Shared canonicalizer (lib/common.py). Repo root on sys.path so this
# stdlib-only helper imports without a PEP 723 dependency entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.common import DriftGate, normalize_deep

# created_at is an immutable creation timestamp. For a model we already
# represent, keep the committed value when the API's value is within this
# window (absorbs a harmless reformat and keeps the created_at ordering stable);
# a jump beyond it fails the run for review, since it shouldn't happen.
CREATED_DRIFT_SECONDS = 24 * 3600

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "anthropic" / "models.json"


def fetch_models() -> list[dict]:
    """Return every model from the Models API (the SDK auto-paginates).

    Credential resolution is delegated to the SDK's zero-arg chain (WIF env
    vars in CI, `ant auth login` profile locally).
    """
    client = anthropic.Anthropic()
    # to_json() serializes datetimes (created_at) to ISO-8601 strings.
    return [json.loads(m.to_json()) for m in client.models.list(limit=100)]


def vanished_ids(models: list[dict], known: set[str]) -> list[str]:
    """Committed ids the API no longer returns (would mutate the file).

    New models are fine — they slot in by created_at and appear as pure
    additions. But a model we already represent disappearing would drop it from
    the file, so that fails the run for human review.
    """
    return sorted(known - {m.get("id", "") for m in models})


def _parse_created(value) -> datetime.datetime | None:
    """Parse an ISO-8601 created_at (tolerating a trailing Z); None if unparseable."""
    try:
        return datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def reconcile_created(models: list[dict], committed: dict[str, str]) -> list[str]:
    """Pin created_at to the committed value; return ids that jumped > 24h.

    created_at should be immutable, so for a model we already represent we keep
    the committed value when the API's is within CREATED_DRIFT_SECONDS — that
    absorbs a harmless reformat and keeps ordering stable. A larger move is
    collected so the caller can fail for human review. Mutates `models`.
    """
    drifted: list[str] = []
    for m in models:
        old = committed.get(m.get("id", ""))
        if old is None:
            continue  # new model — nothing committed to pin against
        parsed_old, parsed_new = _parse_created(old), _parse_created(m.get("created_at"))
        if parsed_old is None or parsed_new is None:
            continue
        if abs((parsed_new - parsed_old).total_seconds()) > CREATED_DRIFT_SECONDS:
            drifted.append(m.get("id", ""))
        else:
            m["created_at"] = old  # pin to the committed value
    return sorted(drifted)


def order_by_created(models: list[dict]) -> dict[str, dict]:
    """Map of id -> model, ordered oldest-first by created_at (id ties)."""
    def sort_key(m: dict):
        created = m.get("created_at")
        return (created is None, str(created), m.get("id", ""))

    return {m.get("id", ""): m for m in sorted(models, key=sort_key)}


def flatten_supported(node: dict, prefix: str = "") -> list[str]:
    """Collapse the capabilities tree to a flat list of supported paths.

    The Models API reports capabilities as a nested tree of {"supported": bool}
    nodes, some with sub-feature children (e.g. effort/low..max,
    thinking/types/adaptive). We keep only the "/"-joined paths flagged
    supported and drop the rest — absence encodes "not supported" — so the blob
    becomes a compact sorted list and daily diffs show one line per capability
    gained or lost. Structural wrappers without their own "supported" flag (e.g.
    thinking/types) never appear as standalone entries, only as path segments.
    """
    paths: list[str] = []
    for key, value in node.items():
        if key == "supported" or not isinstance(value, dict):
            continue
        path = f"{prefix}{key}"
        if value.get("supported") is True:
            paths.append(path)
        paths.extend(flatten_supported(value, prefix=f"{path}/"))
    return paths


# Leading keys for each model, in the order they should appear; any remaining
# keys follow alphabetically.
MODEL_KEY_ORDER = ("id", "display_name", "created_at")


def order_model_keys(model: dict) -> dict:
    """Order a model's top-level keys (preferred first, then A-Z), deep-sorted.

    `capabilities` is collapsed from the API's nested {"supported": bool} tree to
    a flat sorted list of supported paths (see flatten_supported).
    """
    rest = sorted(k for k in model if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in model] + rest
    out: dict = {}
    for k in keys:
        if k == "capabilities" and isinstance(model[k], dict):
            out[k] = sorted(flatten_supported(model[k]))
        else:
            out[k] = normalize_deep(model[k])
    return out


def read_existing_ids(path: Path) -> set[str]:
    """Return the model ids already in the committed snapshot ({} if absent)."""
    if not path.exists():
        return set()
    try:
        with path.open(encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError):
        return set()


def read_existing_created(path: Path) -> dict[str, str]:
    """Map of committed model id -> created_at ({} if the file is absent)."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return {mid: m.get("created_at") for mid, m in json.load(f).items()}
    except (json.JSONDecodeError, OSError, AttributeError):
        return {}


def write_csv(catalog: dict[str, dict], path: Path) -> None:
    """Render the catalog as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["id", "display_name", "created_at", "max_input_tokens", "max_tokens"])
    for mid, m in catalog.items():
        writer.writerow([
            mid,
            m.get("display_name", ""),
            str(m.get("created_at", ""))[:10],
            m.get("max_input_tokens", ""),
            m.get("max_tokens", ""),
        ])
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
    except anthropic.APIStatusError as err:
        print(f"HTTP {err.status_code}\n{err.message}", file=sys.stderr)
        return 1
    except (anthropic.AnthropicError, TypeError) as err:
        # Covers credential-resolution failures (no profile/WIF available);
        # the SDK raises TypeError when no auth method resolves.
        print(f"error: {err}", file=sys.stderr)
        print(
            "hint: run `ant auth login` locally, or set the WIF env vars in CI.",
            file=sys.stderr,
        )
        return 1

    # Drift detection: the committed catalog is the source of truth for what we
    # already represent. New models are free (pure additions), but a model we
    # already have vanishing upstream — or created_at (immutable; pinned to the
    # committed value within 24h) jumping — fails for human review, acknowledged
    # through the DriftGate handshake. Both checks run before failing so one
    # marker records the whole picture and one acknowledgement covers it.
    gate = DriftGate(args.output, allow_all=args.allow_drift, accept_pending=args.accept_pending)
    vanished = gate.unacked("vanished", vanished_ids(models, read_existing_ids(args.output)))
    created_drift = gate.unacked(
        "created_at", reconcile_created(models, read_existing_created(args.output))
    )
    if vanished or created_drift:
        if vanished:
            print(
                f"error: {len(vanished)} committed model(s) no longer returned "
                "by the API:",
                file=sys.stderr,
            )
            for mid in vanished:
                print(f"  {mid}", file=sys.stderr)
        if created_drift:
            print(
                f"error: {len(created_drift)} committed model(s) changed "
                "created_at by more than 24h — it should be immutable:",
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

    print(f"Wrote {len(catalog)} Anthropic models to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
