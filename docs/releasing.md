# Releases

Repository: [zAura-xec/edge-model-bench](https://github.com/zAura-xec/edge-model-bench). Python imports use `edge_model_bench`. The project uses the [MIT License](../LICENSE); preserve its copyright notice when distributing source or binaries.

## Verify the source

```bash
uv sync --extra onnx --frozen
uv run pytest
uv run ruff check src tools examples tests
uv run python -m build
uv run python tools/audit_release.py
```

Install the built wheel into a separate environment and run the generated demo from outside this checkout. Verify both `demo --print-config` and a real CPU evaluation. Configs and the license must be bundled in the wheel.

Before distributing a release, review source/dependency provenance and required notices. SDKs, model weights and datasets retain their own terms and are not bundled. Record hardware verification separately from CPU CI.

## Export a source snapshot

```bash
python tools/prepare_release.py --destination dist/edge-model-bench
```

The exporter includes source, configuration, documentation, tests, examples and selected project files. It honors Git ignore rules and excludes `.git`, virtual environments, local output, models and local configuration. It refuses symlinks and an existing destination. Review any scan findings before proceeding. The archive is a source snapshot without Git history. A zero-finding scan is a heuristic result, not a complete security or provenance review.

When extracting code from a private repository, audit the history separately:

```bash
python tools/audit_release.py --history
```

Do not push private branches, tags or old history to a public repository. Removing current files or changing `.gitignore` does not remove historical blobs. If a credential was committed, handle its revocation privately. The scanner reports object IDs and paths without printing matching values.

For an initial import into an existing public repository, preserve its license and commits: clone the destination, copy the audited snapshot into that clone, inspect the staged diff, and create a normal commit. Push without force; if the remote changes, inspect and reconcile it before retrying.

## Publish

After reviewing the commit, push the intended public branch and verify GitHub Actions. Enable private vulnerability reporting and review the repository description/topics before announcing. Tag a release only after verifying its artifacts. PyPI publication is separate.
