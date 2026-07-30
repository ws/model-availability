#!/usr/bin/env -S uv run --env-file .env --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "google-auth>=2.28",
#     "requests>=2.31",  # required internally by google-auth's WIF/external_account path
# ]
# ///
"""Scrape Google's first-party models from the Gemini API.

Lists models via the Generative Language (Gemini) API:
    GET https://generativelanguage.googleapis.com/v1beta/models

This is the clean, metadata-rich source for Google's own pay-per-token models
(Gemini, Gemma, embeddings, TTS, ...): each entry carries token limits,
supportedGenerationMethods, sampling defaults, etc., and there's no self-deploy
or legacy vision/AutoML noise, so no filtering heuristic is needed.

We use the Gemini API (rather than Vertex Model Garden's `google` publisher) for
Google's first-party models because it carries far richer per-model data (token
limits, supportedGenerationMethods, sampling defaults, thinking flag) and needs
no garden-filtering heuristic, while still covering the current media models
(Imagen, Veo, Lyria). Third-party PAYG models (Claude, Llama-maas, Mistral, ...)
live in list_model_garden_models.py.

Auth is keyless only: ADC scoped to
https://www.googleapis.com/auth/generative-language. User credentials from
`gcloud auth application-default login` can NOT mint this scope (gcloud's OAuth
client isn't authorized for it — you'd get ACCESS_TOKEN_SCOPE_INSUFFICIENT), but
a service account can. So:
  - CI: the same Workload Identity Federation service account used by the Model
    Garden scraper.
  - Local: impersonate that service account —
      gcloud auth application-default login \
        --impersonate-service-account=<scraper-sa>@<project>.iam.gserviceaccount.com
    (requires roles/iam.serviceAccountTokenCreator on the SA).

The snapshot is written to data/providers/google/gemini_api_models.json as a
JSON object keyed by bare model id (the "models/" prefix is dropped from the
key but kept in each value's `name`; the API is global, so there's no region
dimension):

    {"<id>": {"name": "models/<id>", ...}, ...}

Ordering is deterministic and reproducible from scratch: models are emitted in
the hardcoded MODEL_ORDER list below (roughly oldest->newest). The order depends
only on the API response plus that list, so a fresh clone produces the identical
order. Any model not in the list is appended at the bottom (sorted by id) — add
new ids to the end of MODEL_ORDER to place them.

The shebang loads a .env file (uv's built-in --env-file) from the current
working directory, so run this from the repo root.
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

BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
PAGE_SIZE = 200
MAX_PAGES = 100

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = REPO_ROOT / "data" / "providers" / "google" / "gemini_api_models.json"

# Hardcoded catalog order (roughly oldest -> newest). This is the source of
# truth for ordering: models are emitted in this order, and any model not listed
# here is appended at the bottom (sorted by id). Add new ids at the end.
#
# TODO(cleanup): retire this list once the catalog stabilizes. Long term, the
# committed JSON's own order should be the single source of truth (read it, keep
# that order, append new models) — no need for MODEL_ORDER too. Kept hardcoded
# for now only because fresh regenerations during development would lose the
# order otherwise.
# Models returned by the API that we deliberately don't represent. `aqa` is the
# Attributed Question Answering model (generateAnswer only), not a general
# generative model — it's filtered out, and its absence never counts as drift.
EXCLUDE = frozenset({"aqa"})

MODEL_ORDER = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
    "gemini-embedding-001",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "gemini-2.5-flash-image",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-robotics-er-1.5-preview",
    "gemini-2.5-computer-use-preview-10-2025",
    "veo-3.1-fast-generate-preview",
    "veo-3.1-generate-preview",
    "gemini-3-pro-preview",
    "gemini-3-pro-image-preview",
    "imagen-4.0-ultra-generate-001",
    "nano-banana-pro-preview",
    "deep-research-pro-preview-12-2025",
    "gemini-2.5-flash-native-audio-latest",
    "gemini-2.5-flash-native-audio-preview-09-2025",
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-embedding-2-preview",
    "gemini-3.1-flash-live-preview",
    "lyria-3-clip-preview",
    "lyria-3-pro-preview",
    "veo-3.1-lite-generate-preview",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gemini-robotics-er-1.6-preview",
    "gemini-3.1-flash-tts-preview",
    "deep-research-max-preview-04-2026",
    "deep-research-preview-04-2026",
    "gemini-embedding-2",
    "gemini-pro-latest",
    "gemini-3.1-flash-lite",
    "antigravity-preview-05-2026",
    "gemini-3.5-flash",
    "gemini-3-pro-image",
    "gemini-3.1-flash-image",
    "gemini-3.5-live-translate-preview",
    "gemini-3.1-flash-lite-image",
    "gemini-omni-flash-preview",
)


GENERATIVE_LANGUAGE_SCOPE = "https://www.googleapis.com/auth/generative-language"


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


def adc_headers() -> dict:
    """Auth headers from ADC scoped for this API (keyless).

    Only works with service-account credentials (WIF in CI, impersonation
    locally); user credentials can't mint the generative-language scope.
    """
    credentials, adc_project = google.auth.default(scopes=[GENERATIVE_LANGUAGE_SCOPE])
    credentials.refresh(_UrllibTransport())
    headers = {"Authorization": f"Bearer {credentials.token}"}
    quota_project = os.environ.get("GOOGLE_CLOUD_PROJECT") or adc_project
    if quota_project:
        headers["x-goog-user-project"] = quota_project
    return headers


def fetch_models() -> list[dict]:
    """Return every model from the Gemini API, following pagination."""
    headers = adc_headers()

    models: list[dict] = []
    page_token: str | None = None
    for _ in range(MAX_PAGES):
        params = {"pageSize": PAGE_SIZE}
        if page_token:
            params["pageToken"] = page_token
        body = get_json(BASE_URL, params, headers)
        models.extend(body.get("models", []))
        page_token = body.get("nextPageToken")
        if not page_token:
            break
    else:
        print(
            f"warning: stopped after {MAX_PAGES} pages; results may be truncated.",
            file=sys.stderr,
        )
    return models


def model_key(model: dict) -> str:
    """The bare model id used as the catalog key (drops the "models/" prefix)."""
    return model.get("name", "").removeprefix("models/")


def order_by_list(models: list[dict]) -> dict[str, dict]:
    """Map of id -> model, ordered by MODEL_ORDER; unlisted ids appended by id.

    The order is a pure function of the fetched models plus the hardcoded
    MODEL_ORDER, so it's identical on any machine with no dependence on prior
    output.
    """
    index = {mid: i for i, mid in enumerate(MODEL_ORDER)}

    def sort_key(model: dict):
        mid = model_key(model)
        return (0, index[mid]) if mid in index else (1, mid)

    return {model_key(m): m for m in sorted(models, key=sort_key)}


def read_existing_ids(path: Path) -> set[str]:
    """Return the model ids already present in the committed snapshot.

    This is the drift baseline: the committed catalog is the source of truth
    for what we already represent. Missing/corrupt file -> set().
    """
    if not path.exists():
        return set()
    try:
        with path.open(encoding="utf-8") as f:
            return set(json.load(f))
    except (json.JSONDecodeError, OSError):
        return set()


def vanished_ids(models: list[dict], known: set[str]) -> list[str]:
    """Committed ids the API no longer returns (would mutate the existing set).

    New models are fine — they append at the bottom and can't disturb anything.
    But a model we already represent disappearing would drop it from the file
    and shift everything after it, so that fails the run for human review.
    """
    return sorted(known - {model_key(m) for m in models})


# Leading keys for each model, in the order they should appear; any remaining
# keys follow alphabetically.
MODEL_KEY_ORDER = ("name", "displayName", "description", "version")


def order_model_keys(model: dict) -> dict:
    """Order a model's top-level keys (preferred first, then A-Z), deep-sorted."""
    rest = sorted(k for k in model if k not in MODEL_KEY_ORDER)
    keys = [k for k in MODEL_KEY_ORDER if k in model] + rest
    return {k: normalize_deep(model[k]) for k in keys}


def write_csv(catalog: dict[str, dict], path: Path) -> None:
    """Render the catalog as a CSV (GitHub renders it; appends diff cleanly)."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["id", "displayName", "description"])
    for mid, m in catalog.items():
        writer.writerow([
            mid,
            m.get("displayName", ""),
            m.get("description", ""),
        ])
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-drift",
        action="store_true",
        help="Accept removal of committed models instead of erroring",
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
    except urllib.error.HTTPError as err:
        status, body = err.code, err.read().decode()
        print(f"HTTP {status}\n{body}", file=sys.stderr)
        if status == 403 and "SCOPE" in body.upper():
            print(
                "hint: user ADC can't mint the generative-language scope — use "
                "service-account credentials (WIF in CI, impersonation locally; "
                "see the module docstring).",
                file=sys.stderr,
            )
        return 1
    except Exception as err:  # ADC resolution failures (DefaultCredentialsError etc.)
        print(f"error: {err}", file=sys.stderr)
        print(
            "hint: provide service-account ADC — WIF in CI, or local "
            "impersonation (see the module docstring).",
            file=sys.stderr,
        )
        return 1

    # Drop models we deliberately don't represent before anything else, so they
    # never reach the catalog and their absence isn't mistaken for drift.
    models = [m for m in models if model_key(m) not in EXCLUDE]

    # Drift detection: the committed catalog is the source of truth for what we
    # already represent. New models are free (they append at the bottom), but a
    # model we already have vanishing upstream would drop it from the file, so
    # that fails the run for human review.
    drift = vanished_ids(models, read_existing_ids(args.output) - EXCLUDE)
    if drift and not args.allow_drift:
        print(
            f"error: {len(drift)} committed model(s) no longer returned by the "
            "API — review and re-run with --allow-drift to accept the removal:",
            file=sys.stderr,
        )
        for mid in drift:
            print(f"  {mid}", file=sys.stderr)
        return 1

    # Emit in the hardcoded MODEL_ORDER. Key by the bare model id, with the full
    # resource path preserved in each value's `name`.
    ordered = order_by_list(models)
    catalog = {mid: order_model_keys(model) for mid, model in ordered.items()}

    # Ordering is intentional, so no sort_keys; inner keys are already
    # normalized by order_model_keys/normalize_deep.
    payload = json.dumps(catalog, indent=2, ensure_ascii=False)

    # Write atomically so a failure can never leave a truncated snapshot behind.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(args.output)

    write_csv(catalog, args.output.with_suffix(".csv"))

    print(f"Wrote {len(catalog)} Gemini API models to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
