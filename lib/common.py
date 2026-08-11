"""Shared, dependency-free helpers for the provider catalog scrapers.

Kept stdlib-only so the uv single-file scripts (PEP 723) can import it without
declaring it as a dependency — each script puts the repo root on sys.path and
does `from lib.common import ...`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


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


class DriftGate:
    """Two-phase drift acknowledgement: record on failure, accept on re-run.

    Immutable-field drift fails the run for human review. The acknowledgement
    has to be possible from the
    GitHub mobile app, which can re-run a failed workflow but can't pass
    flags — so the failing run records exactly the drift it saw in a marker
    file next to the snapshot (`<snapshot>.pending-drift.json`, committed by
    the workflow), and a run with `accept_pending` set (the workflows pass
    --accept-pending on re-runs and manual dispatches) accepts any drift
    covered by that marker, deleting it in the same commit that applies the
    change. Drift the marker doesn't cover fails again and re-records, so an
    acknowledgement can never accept more than what was already reported.
    Scheduled runs never pass `accept_pending`, so a pending marker keeps
    failing (and notifying) daily until a human re-runs.

    Usage, per drift kind:

        gate = DriftGate(args.output, allow_all=args.allow_drift,
                         accept_pending=args.accept_pending)
        drift = gate.unacked("created", changed_ids(...))
        if drift:
            ...print the ids...
            gate.record()
            return 1
        ...write the snapshot...
        gate.clear()
    """

    def __init__(self, output: Path, allow_all: bool = False, accept_pending: bool = False):
        self.path = output.with_name(output.stem + ".pending-drift.json")
        self.allow_all = allow_all
        self.accept_pending = accept_pending
        self.seen: dict[str, list[str]] = {}
        self.checked: set[str] = set()
        try:
            recorded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            recorded = {}
        self.recorded: dict = recorded if isinstance(recorded, dict) else {}

    def unacked(self, kind: str, ids: list[str]) -> list[str]:
        """Note drift of `kind`; return the ids not covered by an acknowledgement.

        Everything is covered under allow_all; under accept_pending, ids the
        marker already records are covered (the drift a human saw and re-ran
        to accept); otherwise any drift is unacknowledged.
        """
        ids = sorted(ids)
        self.checked.add(kind)
        if ids:
            self.seen[kind] = ids
        if self.allow_all:
            return []
        if not self.accept_pending:
            return ids
        recorded = set(self.recorded.get(kind, []))
        return [i for i in ids if i not in recorded]

    def record(self) -> None:
        """Write the marker for this run's drift and explain the handshake.

        Kinds this run never checked (e.g. another region's pending record)
        are carried over untouched — a run only speaks for what it looked at.
        """
        carried = {k: v for k, v in self.recorded.items() if k not in self.checked}
        payload = json.dumps({**carried, **self.seen}, indent=2, ensure_ascii=False)
        self.path.write_text(payload + "\n", encoding="utf-8")
        print(
            f"note: drift recorded in {self.path} — if it's expected, "
            "re-run the failed workflow (Re-run failed jobs; the GitHub "
            "mobile app can) or re-run locally with --accept-pending to "
            "accept exactly this drift. Different drift fails again.",
            file=sys.stderr,
        )

    def clear(self) -> None:
        """Drop the checked kinds from the marker: applied, or healed upstream.

        Kinds this run never checked are kept; the file is deleted once
        nothing pending remains.
        """
        carried = {k: v for k, v in self.recorded.items() if k not in self.checked}
        if carried:
            payload = json.dumps(carried, indent=2, ensure_ascii=False)
            self.path.write_text(payload + "\n", encoding="utf-8")
        else:
            self.path.unlink(missing_ok=True)
