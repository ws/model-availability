#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Scrape pay-as-you-go (MaaS) models from the Azure AI Foundry model catalog.

Queries the public asset-gallery API that backs ai.azure.com/catalog:
    POST https://api.catalog.azureml.ms/asset-gallery/v1.0/models
The API is anonymous — no credentials or subscription needed.

Scope: we keep only models billed on a marginal/pay-per-token basis, using the
server-side filter deploymentOptions IN (Serverless API, AOAI,
UnifiedEndpointMaaS). That excludes "Managed compute" (deploy-your-own, billed
for provisioned VMs). "Instant" is a subset of the above and adds nothing.

The API returns one summary per model *version*; we keep only the latest
version per model name (highest createdTime), matching how the catalog UI
presents models. `popularity` (a constantly-shifting ranking float) and
`createdTime` (a server timestamp the API re-stamps per request) are dropped
from the output so daily diffs only show real catalog changes.

The snapshot is written to data/providers/azure/foundry_models.json as a JSON
object keyed by model name (the catalog's identity — also its URL slug at
ai.azure.com/catalog/models/<name>); the catalog is global, so there's no
region dimension (per-region availability lives in each model's deploymentSku
locations). Models are ordered by name (the stable catalog key) — createdTime
can't be used for ordering because it drifts between scrapes. A sibling
foundry_models.csv is written for human review.

"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Shared canonicalizer (lib/common.py). Repo root on sys.path so this
# stdlib-only helper imports without a PEP 723 dependency entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.common import normalize_deep

API_URL = "https://api.catalog.azureml.ms/asset-gallery/v1.0/models"
# Marginal/pay-per-token deployment options (multi-value filter = OR).
MAAS_DEPLOYMENT_OPTIONS = ("Serverless API", "AOAI", "UnifiedEndpointMaaS")
PAGE_SIZE = 100
MAX_PAGES = 100

# Keys dropped from the output. popularity is a ranking float that shifts every
# scrape and would swamp the diffs with noise; deploymentSku is a huge per-SKU
# region matrix that isn't worth the space/churn; createdTime is a volatile
# server timestamp (the API re-stamps it per request — the same model's value
# drifts by seconds to a full year between scrapes), so it's pure churn and
# useless as a stable creation date; labels are catalog-UI hints ("latest",
# "invisibleLatest", "default") that carry no availability signal — every
# labeled model has latest/invisibleLatest and "default" flickers on/off
# between scrapes, so they're pure churn (real status lives in `lifecycle`).
DROP_KEYS = {"popularity", "deploymentSku", "createdTime", "labels"}

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "azure" / "foundry_models.json"


def post_json(url: str, payload: dict) -> dict:
    """POST a JSON body and return the parsed JSON response (stdlib only)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch_maas_models() -> list[dict]:
    """Return every MaaS model-version summary, following pagination.

    The API caps pageSize at 100, so the full catalog (~240 version summaries)
    spans several requests. Pagination is plain `$skip` offset paging over
    Azure Cognitive Search (the continuationToken decodes to {"NextSkip": N}),
    executed as separate HTTP requests seconds apart. The default sort is by
    popularity — a value that reshuffles between requests — so a model whose
    rank crosses a page boundary (skip=100, 200, ...) between requests gets
    skipped entirely, showing up as a spurious removal. Sorting by `name`
    (stable across requests) pins the page boundaries so offset paging stays
    gap-free; the only way to lose a model now is a genuine catalog change
    mid-scrape.
    """
    body = {
        "filters": [
            {
                "field": "deploymentOptions",
                "values": list(MAAS_DEPLOYMENT_OPTIONS),
                "operator": "eq",
            }
        ],
        "order": [{"field": "name", "direction": "asc"}],
        "pageSize": PAGE_SIZE,
    }
    models: list[dict] = []
    continuation: str | None = None
    for _ in range(MAX_PAGES):
        payload = dict(body)
        if continuation:
            payload["continuationToken"] = continuation
        page = post_json(API_URL, payload)
        models.extend(page.get("summaries", []))
        continuation = page.get("continuationToken")
        if not continuation:
            break
    else:
        print(
            f"warning: stopped after {MAX_PAGES} pages; results may be truncated.",
            file=sys.stderr,
        )
    return models


def canonical(name: str) -> str:
    """Case-insensitive identity for a model name.

    Azure sometimes returns the same model under different capitalization
    (e.g. gpt-oss-120B vs gpt-oss-120b). Those are one model, not two, so we
    compare names case-insensitively everywhere identity matters (dedup, drift).
    The written key keeps a stable casing — see apply_committed_casing.
    """
    return name.casefold()


def latest_versions(models: list[dict]) -> list[dict]:
    """Collapse per-version summaries to the latest version per model.

    Grouped by case-insensitive name (see canonical), so a model returned under
    two capitalizations collapses to one entry instead of duplicating.

    createdTime is used only to pick which version to keep (the best available
    recency signal — version strings are heterogeneous: numbers, dates,
    "1106-Preview"). It's dropped from the emitted output; in practice the pick
    is stable and flips only when a genuinely newer version appears.
    """
    best: dict[str, dict] = {}
    for m in models:
        cid = canonical(m.get("name", ""))
        if cid not in best or str(m.get("createdTime", "")) > str(
            best[cid].get("createdTime", "")
        ):
            best[cid] = m
    return list(best.values())


def apply_committed_casing(models: list[dict], known: set[str]) -> None:
    """Pin each model's name to the casing already committed, if any.

    Identity is case-insensitive, but the written key should be stable: once
    gpt-oss-120b is committed, keep that casing even when the API later returns
    gpt-oss-120B. Brand-new models keep whatever casing the API first gave.
    Mutates each model's `name` in place. `sorted` makes the pick deterministic
    if the committed catalog itself still holds a case-collision.
    """
    committed = {canonical(n): n for n in sorted(known)}
    for m in models:
        pinned = committed.get(canonical(m.get("name", "")))
        if pinned is not None:
            m["name"] = pinned


def apply_committed_asset_ids(models: list[dict], committed: dict[str, dict]) -> None:
    """Pin each model's assetId to the committed value on case-only flicker.

    Azure re-cases the model-name segment of assetId between scrapes
    (azureml://.../models/Flux-1.1-Pro/versions/1 vs .../FLUX-1.1-pro/...) with
    no real change — and that segment doesn't track the model's own `name`
    casing, so it can't be derived. When the incoming assetId differs from the
    committed one by case alone, keep the committed value; a genuine change
    (different registry, version, or slug) still passes through unchanged.
    Matched by case-insensitive name, so it survives the name re-casing above.
    """
    prior = {canonical(n): m.get("assetId") for n, m in committed.items()}
    for m in models:
        old = prior.get(canonical(m.get("name", "")))
        new = m.get("assetId")
        if old and isinstance(new, str) and old.casefold() == new.casefold():
            m["assetId"] = old


def order_by_name(models: list[dict]) -> dict[str, dict]:
    """Map of name -> model, ordered by name (the stable catalog key).

    Ordering by name — not createdTime — because createdTime is re-stamped by
    the API per request, so a createdTime order reshuffles the whole file every
    scrape even when nothing changed. name is the identity and never drifts.
    """
    return {m.get("name", ""): m for m in sorted(models, key=lambda m: m.get("name", ""))}


# Leading keys for each model, in the order they should appear; any remaining
# keys follow alphabetically.
MODEL_KEY_ORDER = (
    "name",
    "displayName",
    "publisher",
    "registryName",
    "version",
    "lifecycle",
    "deploymentOptions",
)


def order_model_keys(model: dict) -> dict:
    """Order a model's top-level keys (preferred first, then A-Z), deep-sorted.

    Keys in DROP_KEYS are omitted from the output.
    """
    present = [k for k in model if k not in DROP_KEYS]
    rest = sorted(k for k in present if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in present] + rest
    return {k: normalize_deep(model[k]) for k in keys}


def read_existing_catalog(path: Path) -> dict[str, dict]:
    """Return the committed snapshot as name -> model ({} if absent/unreadable)."""
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_csv(catalog: dict[str, dict], path: Path) -> None:
    """Render the catalog as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["name", "displayName", "publisher", "lifecycle"])
    for name, m in catalog.items():
        writer.writerow([
            name,
            m.get("displayName", ""),
            m.get("publisher", ""),
            m.get("lifecycle", ""),
        ])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the JSON snapshot (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        raw = fetch_maas_models()
    except urllib.error.HTTPError as err:
        print(f"HTTP {err.code} {err.reason}\n{err.read().decode()}", file=sys.stderr)
        return 1

    models = latest_versions(raw)

    # Keep the committed casing for models we already represent, so a case-only
    # change upstream doesn't rewrite the key (identity is case-insensitive).
    existing = read_existing_catalog(args.output)
    apply_committed_casing(models, set(existing))
    apply_committed_asset_ids(models, existing)

    catalog = order_by_name(models)
    catalog = {name: order_model_keys(m) for name, m in catalog.items()}

    # Name ordering is intentional, so no sort_keys; inner keys are already
    # normalized by order_model_keys/normalize_deep.
    payload = json.dumps(catalog, indent=2, ensure_ascii=False)

    # Write atomically so a failure can never leave a truncated snapshot behind.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(args.output)

    write_csv(catalog, args.output.with_suffix(".csv"))

    publishers = {m.get("publisher") for m in catalog.values()}
    print(
        f"Wrote {len(catalog)} Foundry MaaS models from {len(publishers)} "
        f"publishers to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
