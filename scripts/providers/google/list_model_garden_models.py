#!/usr/bin/env -S uv run --env-file .env --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-auth>=2.28",
#     "requests>=2.31",  # required internally by google-auth's WIF/external_account path
# ]
# ///
"""Scrape third-party pay-as-you-go (MaaS) models from Google Cloud Model Garden.

Lists publisher models via the Vertex AI REST API (the endpoint is still
aiplatform.googleapis.com even as the product branding moves to "Google"):
    GET https://{region}-aiplatform.googleapis.com/v1beta1/publishers/{publisher}/models

Discovery: the `publishers/*` wildcard parent lists models across *every*
publisher in one paginated sweep, so we don't maintain a publisher list by hand
and new publishers are picked up automatically on each run. (The AIP-standard
`publishers/-` wildcard returns nothing here; the console's GraphQL backend uses
`publishers/*`.) The regional endpoint has much broader coverage than the global
one, so we query per region and paginate via nextPageToken. We also apply the
server-side `is_hf_wildcard(false)` filter (the same one the console uses) to
drop the ~11.7k of ~14k mirrored Hugging Face community uploads.

Scope: we keep only third-party models billed on a marginal/pay-per-token basis
(MaaS), which means we DROP:
  - Google's own first-party models — those are covered by the Gemini API in
    list_gemini_api_models.py (a cleaner source, no legacy-garden noise).
  - self-deploy models (supportedActions has deploy/deployGke/multiDeployVertex)
    — those bill for provisioned compute, not marginal usage.
A model is treated as MaaS/PAYG when it isn't self-deploy AND carries a managed
signal (a `-maas` name, an openGenerationAiStudio/requestAccess action, or a
publisherModelTemplate). There is no authoritative MaaS flag in the API, so this
is a heuristic; pass --all to emit the full unfiltered catalog for inspection.

The snapshot is written to data/providers/google/model_garden_models.json as a
JSON object keyed by region, each region an object keyed by the model's resource
name:

    {"us-central1": {"<pub>/<id>": {"name": "publishers/<pub>/models/<id>", ...}}}

Each run scrapes one region and merges its result into the existing file,
leaving other regions untouched. Region keys are sorted. Within a region, models
are emitted in the hardcoded MODEL_ORDER list below (roughly oldest->newest, by
full resource name). The order depends only on the API response plus that list,
so a fresh clone produces the identical order. Any model not in the list is
appended at the bottom (sorted by name) — add new names to the end to place them.

Auth uses Application Default Credentials (google.auth.default). For local runs:
    gcloud auth application-default login
In CI, prefer Workload Identity Federation (the GCP analog of the AWS OIDC role)
so no service-account key is stored.

The shebang loads a .env file (uv's built-in --env-file) from the current
working directory, so run this from the repo root. Region/project/publishers
default below and can be overridden by flags or the matching env vars.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import urllib.error
import urllib.parse
import urllib.request

import google.auth
import google.auth.exceptions
import google.auth.transport

# Shared canonicalizer (lib/common.py). Repo root on sys.path so this
# stdlib-only helper imports without a PEP 723 dependency entry.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from lib.common import normalize_deep

DEFAULT_REGION = "us-central1"
# Wildcard parent that lists models across every publisher (the console's
# GraphQL backend uses "*", not the AIP-standard "-").
WILDCARD_PUBLISHER = "*"
# Publishers dropped entirely: Google's own (covered by the Gemini API scraper)
# and internal Google test artifacts (e.g. multi-region-test-model).
EXCLUDED_PUBLISHERS = {"google", "internal-test-google"}
# Presence of any of these actions means the model must be deployed to a
# provisioned endpoint (billed for compute), i.e. not marginal/pay-per-token.
DEPLOY_ACTIONS = {"deploy", "deployGke", "multiDeployVertex"}
SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
PAGE_SIZE = 200
# Backstop against an unexpected pagination loop (~70 pages of 200 today).
MAX_PAGES = 500

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "google" / "model_garden_models.json"

# Hardcoded catalog order (roughly oldest -> newest), by '<publisher>/<id>' key.
# This is the source of truth for ordering: models are emitted in this order,
# and any model not listed here is appended at the bottom (sorted by key). Add
# new keys at the end.
#
# TODO(cleanup): retire this list once the catalog stabilizes. Long term, the
# committed JSON's own order should be the single source of truth (read it, keep
# that order, append new models) — no need for MODEL_ORDER too. Kept hardcoded
# for now only because fresh regenerations during development would lose the
# order otherwise.
MODEL_ORDER = (
    "meta/llama-3.3-70b-instruct-maas",
    "mistralai/mistral-small-2503",
    "qwen/qwen3-coder-480b-a35b-instruct-maas",
    "meta/llama-4-maverick-17b-128e-instruct-maas",
    "mistralai/mistral-medium-3",
    "mistralai/mistral-ocr-2505",
    "deepseek-ai/deepseek-r1-0528-maas",
    "qwen/qwen3-235b-a22b-instruct-2507-maas",
    "anthropic/claude-opus-4-1",
    "openai/gpt-oss-120b-maas",
    "deepseek-ai/deepseek-v3.1-maas",
    "qwen/qwen3-next-80b-a3b-instruct-maas",
    "qwen/qwen3-next-80b-a3b-thinking-maas",
    "intfloat/multilingual-e5-large-instruct-maas",
    "anthropic/claude-sonnet-4-5",
    "anthropic/claude-haiku-4-5",
    "mistralai/codestral-2",
    "deepseek-ai/deepseek-ocr-maas",
    "minimaxai/minimax-m2-maas",
    "moonshotai/kimi-k2-thinking-maas",
    "xai/grok-4.1-fast-non-reasoning",
    "xai/grok-4.1-fast-reasoning",
    "anthropic/claude-opus-4-5",
    "deepseek-ai/deepseek-v3.2-maas",
    "zai-org/glm-4.7-maas",
    "anthropic/claude-opus-4-6",
    "zai-org/glm-5-maas",
    "anthropic/claude-sonnet-4-6",
    "xai/grok-4.20-non-reasoning",
    "xai/grok-4.20-reasoning",
    "anthropic/claude-opus-4-7",
    "xai/grok-4.3",
    "anthropic/claude-opus-4-8",
    "anthropic/claude-fable-5",
    "anthropic/claude-sonnet-5",
)


class _UrllibResponse(google.auth.transport.Response):
    """google.auth transport response backed by stdlib urllib."""

    def __init__(self, status: int, headers: dict, data: bytes):
        self._status, self._headers, self._data = status, headers, data

    @property
    def status(self) -> int:
        return self._status

    @property
    def headers(self) -> dict:
        return self._headers

    @property
    def data(self) -> bytes:
        return self._data


class _UrllibTransport(google.auth.transport.Request):
    """Minimal stdlib transport for credential refresh (avoids the requests dep).

    Implements google-auth's documented Request ABC; used only for token
    fetch/exchange during credentials.refresh().
    """

    def __call__(self, url, method="GET", body=None, headers=None, timeout=60, **kwargs):
        if isinstance(body, str):
            body = body.encode()
        request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return _UrllibResponse(resp.status, dict(resp.headers), resp.read())
        except urllib.error.HTTPError as exc:
            return _UrllibResponse(exc.code, dict(exc.headers), exc.read())
        except Exception as exc:
            raise google.auth.exceptions.TransportError(exc) from exc


def get_json(url: str, params: dict, headers: dict) -> dict:
    """GET a JSON document (stdlib only); raises urllib.error.HTTPError on 4xx/5xx."""
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full_url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def adc_headers(project: str | None) -> dict:
    """Return auth headers from ADC, with the quota project set."""
    credentials, adc_project = google.auth.default(scopes=SCOPES)
    credentials.refresh(_UrllibTransport())

    headers = {"Authorization": f"Bearer {credentials.token}"}
    quota_project = project or adc_project
    if quota_project:
        # Bills the list call and satisfies the API's quota-project requirement.
        headers["x-goog-user-project"] = quota_project
    return headers


def publisher_of(model: dict) -> str:
    """Extract the publisher slug from a model's resource name."""
    name = model.get("name", "")
    parts = name.split("/")
    return parts[1] if len(parts) > 1 and parts[0] == "publishers" else ""


def model_key(model: dict) -> str:
    """Catalog key '<publisher>/<id>' (drops the publishers/ and models/ parts).

    The full resource path is still preserved in each value's `name` field.
    """
    return model.get("name", "").removeprefix("publishers/").replace("/models/", "/", 1)


def is_third_party_payg(model: dict) -> bool:
    """Heuristic: a third-party model billed on a marginal/pay-per-token basis.

    Excludes Google's own models (covered by the Gemini API scraper) and any
    self-deploy model (billed for provisioned compute). Among what's left, we
    require a positive "managed serving" signal, since the API has no explicit
    MaaS flag. The self-deploy exclusion keeps the legacy vision/AutoML garden —
    which is Google-first-party anyway — out of the third-party result.
    """
    if publisher_of(model) in EXCLUDED_PUBLISHERS:
        return False
    actions = set(model.get("supportedActions", {}))
    if actions & DEPLOY_ACTIONS:
        return False
    return (
        model.get("name", "").endswith("-maas")
        or "openGenerationAiStudio" in actions
        or "requestAccess" in actions
        or "publisherModelTemplate" in model
    )


def fetch_all_models(headers: dict, region: str, include_hf: bool) -> list[dict]:
    """Return every publisher model in a region via the `publishers/*` wildcard.

    Unless include_hf is set, applies the server-side `is_hf_wildcard(false)`
    filter (the same one the Model Garden console uses) to drop the thousands of
    mirrored Hugging Face community uploads before they're paged over.
    """
    url = (
        f"https://{region}-aiplatform.googleapis.com"
        f"/v1beta1/publishers/{WILDCARD_PUBLISHER}/models"
    )
    base_params = {"pageSize": PAGE_SIZE}
    if not include_hf:
        base_params["filter"] = "is_hf_wildcard(false)"

    models: list[dict] = []
    page_token: str | None = None
    for _ in range(MAX_PAGES):
        params = dict(base_params)
        if page_token:
            params["pageToken"] = page_token
        body = get_json(url, params, headers)
        models.extend(body.get("publisherModels", []))
        page_token = body.get("nextPageToken")
        if not page_token:
            break
    else:
        print(
            f"warning: stopped after {MAX_PAGES} pages; results may be truncated.",
            file=sys.stderr,
        )
    return models


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


def order_by_list(models: list[dict]) -> dict[str, dict]:
    """Map of resource name -> model, ordered by MODEL_ORDER; rest appended.

    The order is a pure function of the fetched models plus the hardcoded
    MODEL_ORDER, so it's identical on any machine with no dependence on prior
    output. Models not in the list are appended at the bottom (sorted by name).
    """
    index = {key: i for i, key in enumerate(MODEL_ORDER)}

    def sort_key(model: dict):
        key = model_key(model)
        return (0, index[key]) if key in index else (1, key)

    return {model_key(m): m for m in sorted(models, key=sort_key)}


def vanished_keys(models: list[dict], known: set[str]) -> list[str]:
    """Committed keys the API no longer returns (would mutate the existing set).

    New models are fine — they append at the bottom and can't disturb anything.
    But a model we already represent disappearing would drop it from the file
    and shift everything after it, so that fails the run for human review.
    """
    return sorted(known - {model_key(m) for m in models})


# Leading keys for each model, in the order they should appear; any remaining
# keys follow alphabetically.
MODEL_KEY_ORDER = ("name", "versionId", "launchStage", "openSourceCategory")
# Keys dropped from the output. supportedActions is only used to classify PAYG
# vs self-deploy (see is_third_party_payg); it's noise in the snapshot.
DROP_KEYS = {"supportedActions"}


def order_model_keys(model: dict) -> dict:
    """Order a model's top-level keys (preferred first, then A-Z), deep-sorted.

    Keys in DROP_KEYS are omitted from the output.
    """
    present = [k for k in model if k not in DROP_KEYS]
    rest = sorted(k for k in present if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in present] + rest
    return {k: normalize_deep(model[k]) for k in keys}


def write_csv(snapshot: dict[str, dict], path: Path) -> None:
    """Render the snapshot as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["region", "model", "launchStage", "versionId", "openSourceCategory"])
    for region, catalog in snapshot.items():
        for key, m in catalog.items():
            writer.writerow([
                region,
                key,
                m.get("launchStage", ""),
                m.get("versionId", ""),
                m.get("openSourceCategory", ""),
            ])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--region",
        default=os.environ.get("GOOGLE_CLOUD_REGION", DEFAULT_REGION),
        help="Vertex AI location (default: %(default)s)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Emit the full unfiltered catalog instead of only third-party PAYG",
    )
    parser.add_argument(
        "--include-hf",
        action="store_true",
        help="Include hf-* publishers (mirrored Hugging Face uploads); off by default",
    )
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Accept removal of committed models instead of erroring",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Quota/billing project (default: $GOOGLE_CLOUD_PROJECT or ADC project)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write the JSON snapshot (default: %(default)s)",
    )
    args = parser.parse_args()

    try:
        headers = adc_headers(args.project)
        models = fetch_all_models(headers, args.region, args.include_hf)
    except urllib.error.HTTPError as err:
        # Surface the response body — invaluable for diagnosing 4xx (disabled
        # API, permission denied, wrong/missing quota project, bad region, ...).
        print(f"HTTP {err.code} {err.reason}\n{err.read().decode()}", file=sys.stderr)
        return 1

    snapshot = read_snapshot(args.output)

    if not args.all:
        models = [m for m in models if is_third_party_payg(m)]

        # Drift detection: the committed catalog is the source of truth for
        # what we already represent in this region. New models are free (they
        # append at the bottom), but a model we already have vanishing upstream
        # would drop it from the file, so that fails for human review. Skipped
        # under --all (that's an unordered inspection dump).
        drift = vanished_keys(models, set(snapshot.get(args.region, {})))
        if drift and not args.allow_drift:
            print(
                f"error: {len(drift)} committed model(s) no longer returned by "
                "the API — review and re-run with --allow-drift to accept the "
                "removal:",
                file=sys.stderr,
            )
            for key in drift:
                print(f"  {key}", file=sys.stderr)
            return 1

    # Emit in the hardcoded MODEL_ORDER.
    catalog = order_by_list(models)
    catalog = {name: order_model_keys(model) for name, model in catalog.items()}

    # Merge this region into the existing snapshot (preserving other regions),
    # then sort region keys so the top level is stable regardless of scrape order.
    snapshot[args.region] = catalog
    snapshot = {region: snapshot[region] for region in sorted(snapshot)}

    # Region and name ordering is intentional, so no sort_keys; inner keys are
    # already normalized by order_model_keys/normalize_deep.
    payload = json.dumps(snapshot, indent=2, ensure_ascii=False)

    # Write atomically so a failure can never leave a truncated snapshot behind.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(args.output)

    write_csv(snapshot, args.output.with_suffix(".csv"))

    publisher_count = len({publisher_of(m) for m in catalog.values()})
    print(
        f"Wrote {len(catalog)} models from {publisher_count} publishers "
        f"for {args.region} to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
