# model-availability

Daily snapshots of the model catalogs offered by the major AI providers, using Simon Willison's [git scraping](https://simonwillison.net/2021/Mar/5/git-scraping/) technique. The git history is the dataset: `git log -p -- data/` shows exactly when a model appeared, changed, or was removed, and each commit message summarizes that day's delta.

## What's tracked

| Provider | Catalog | Snapshot |
| --- | --- | --- |
| AWS Bedrock | Runtime/Foundation models (us-east-1) | [`data/providers/bedrock/foundation_models.json`](data/providers/bedrock/foundation_models.json) |
| AWS Bedrock | Mantle models (us-east-1) | [`data/providers/bedrock/mantle_models.json`](data/providers/bedrock/mantle_models.json) |
| Azure | AI Foundry pay-as-you-go (MaaS) models | [`data/providers/azure/foundry_models.json`](data/providers/azure/foundry_models.json) |
| Google | Gemini API models | [`data/providers/google/gemini_api_models.json`](data/providers/google/gemini_api_models.json) |
| Google | Model Garden third-party PAYG models (us-central1) | [`data/providers/google/model_garden_models.json`](data/providers/google/model_garden_models.json) |
| Anthropic | First-party models | [`data/providers/anthropic/models.json`](data/providers/anthropic/models.json) |
| OpenAI | First-party models | [`data/providers/openai/models.json`](data/providers/openai/models.json) |

Every JSON snapshot has a sibling `.csv` for human review. Scope is deliberately pay-per-token: self-deploy / managed-compute offerings (Azure managed compute, Model Garden deployables, Hugging Face mirrors) are filtered out.

## How it works

One workflow per provider (`.github/workflows/`), daily, staggered an hour apart so pushes never race. Scrapers are single-file [uv](https://docs.astral.sh/uv/) scripts (PEP 723) under `scripts/providers/`; auth is keyless throughout (GitHub OIDC → provider WIF/role; Azure's catalog API is anonymous). A run commits only if something actually changed, with a deterministic message from `scripts/diff_summary.py`.

The design decisions that keep the diffs honest:

- **Canonicalization** (`lib/common.py`): dict keys sorted, unordered arrays sorted, so provider APIs that shuffle response order between requests can't churn the snapshots.
- **Volatile fields dropped**: server-stamped timestamps, popularity rankings, and other per-request noise are stripped so a diff means the catalog really changed.
- **Stable ordering**: models are emitted oldest-to-newest (or by a pinned order list), so new models append at the bottom of the file and diffs stay minimal.
- **Drift detection**: new models are free, but a previously-committed model vanishing upstream fails the run for human review. Azure additionally re-fetches once before trusting a disappearance, since its API occasionally flickers.
- **Drift acknowledgement, phone-friendly**: the failing run commits a `<snapshot>.pending-drift.json` marker recording exactly the drift it saw. Re-running the failed workflow ("Re-run failed jobs", which the GitHub mobile app can do straight from the failure notification) accepts exactly that recorded drift and deletes the marker in the same commit that applies it. Drift the marker doesn't cover fails again and re-records, and scheduled runs never auto-accept, so unacknowledged drift keeps failing daily until a human re-runs. Locally, `--accept-pending` is the same acknowledgement; `--allow-drift` still accepts everything unconditionally.
