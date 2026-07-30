#!/usr/bin/env python3
"""Summarize model-catalog changes for an auto-commit message.

Compares each given JSON snapshot against its committed (HEAD) version and
reports which model ids were added, removed, or modified (and, for modified
ones, which top-level fields changed), rendered as a deterministic commit
message: counts in the subject, ids in the body.

Snapshot shapes handled: flat {id: model} and region-nested {region: {id: model}}
(bedrock foundation). Model ids are the innermost keys either way.

Run from the repo root (paths are resolved against HEAD with `git show`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def committed_version(path: str) -> dict:
    """The file's parsed content at HEAD, or {} if it's new or unreadable."""
    try:
        blob = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return json.loads(blob)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {}


def flatten_models(catalog: dict) -> dict:
    """Return {id: model}, flattening a region-nested snapshot if present.

    A snapshot is either flat {id: model} or region-nested {region: {id: model}}.
    It's treated as region-nested only when every top-level value is a non-empty
    dict whose own values are all dicts (the per-region model maps). A flat
    catalog fails that test because a model always has scalar fields (id,
    created, ...), so its values are not all dicts.
    """
    if not isinstance(catalog, dict) or not catalog:
        return {}
    nested = all(
        isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values())
        for v in catalog.values()
    )
    if not nested:
        return catalog
    flat: dict = {}
    for region in catalog.values():
        flat.update(region)  # same id in >1 region -> identical model; last wins
    return flat


def changed_fields(old: dict, new: dict) -> list[str]:
    """Top-level field names whose value differs between two model dicts."""
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def diff_catalog(path: str) -> dict:
    """Added/removed/modified delta for one snapshot file (working tree vs HEAD)."""
    old = flatten_models(committed_version(path))
    new = flatten_models(json.loads(Path(path).read_text(encoding="utf-8")))
    return {
        "name": Path(path).stem,
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "modified": {
            mid: changed_fields(old[mid], new[mid])
            for mid in sorted(set(old) & set(new))
            if old[mid] != new[mid]
        },
    }


def build_delta(paths: list[str]) -> dict:
    """Per-catalog deltas (empty ones dropped) plus aggregate counts."""
    catalogs = [diff_catalog(p) for p in paths]
    catalogs = [c for c in catalogs if c["added"] or c["removed"] or c["modified"]]
    totals = {
        "added": sum(len(c["added"]) for c in catalogs),
        "removed": sum(len(c["removed"]) for c in catalogs),
        "modified": sum(len(c["modified"]) for c in catalogs),
    }
    return {"catalogs": catalogs, "totals": totals}


def render_message(delta: dict, title: str) -> str:
    """Deterministic commit message: counts in the subject, ids in the body."""
    t = delta["totals"]
    if not delta["catalogs"]:
        return f"Update {title} (no model-level changes)"
    lines = [f"Update {title}: +{t['added']} ~{t['modified']} -{t['removed']}", ""]
    multi = len(delta["catalogs"]) > 1
    for c in delta["catalogs"]:
        if multi:
            lines.append(f"[{c['name']}]")
        if c["added"]:
            lines.append(f"Added: {', '.join(c['added'])}")
        if c["removed"]:
            lines.append(f"Removed: {', '.join(c['removed'])}")
        if c["modified"]:
            mods = ", ".join(
                f"{mid} ({', '.join(fields)})" for mid, fields in c["modified"].items()
            )
            lines.append(f"Modified: {mods}")
        if multi:
            lines.append("")
    return "\n".join(lines).rstrip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="JSON snapshot file(s) for one provider")
    ap.add_argument("--title", default=None, help="Provider label for the subject line")
    args = ap.parse_args()

    title = args.title or f"{Path(args.paths[0]).parent.name} catalog"
    print(render_message(build_delta(args.paths), title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
