"""Shared, dependency-free helpers for the provider catalog scrapers.

Kept stdlib-only so the uv single-file scripts (PEP 723) can import it without
declaring it as a dependency — each script puts the repo root on sys.path and
does `from lib.common import ...`.
"""

from __future__ import annotations

import json


def normalize_deep(obj):
    """Canonicalize a decoded-JSON value for stable, churn-free git diffs.

    - dict keys are sorted;
    - scalar lists are sorted by their string form;
    - lists containing objects (or a mix) are sorted by each element's canonical
      JSON, so a provider API that returns array elements in a different order
      between scrapes can never churn the committed snapshot.

    Array order is treated as insignificant. That holds for these catalogs —
    every array is an unordered set (modalities, tags, options, SKUs,
    languages). If a future field ever carried a meaningful order it would have
    to be exempted here. This is the single canonicalizer for every provider;
    key-order and array-order drift are handled here so the scrapers don't each
    reinvent it (and drift apart).
    """
    if isinstance(obj, dict):
        return {k: normalize_deep(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        items = [normalize_deep(v) for v in obj]
        if all(isinstance(v, (str, int, float, bool)) for v in items):
            return sorted(items, key=str)
        return sorted(items, key=lambda v: json.dumps(v, sort_keys=True, ensure_ascii=False))
    return obj
