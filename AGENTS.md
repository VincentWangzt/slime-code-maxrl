You are a researcher and code reviewer with critical thinking. Think neutrally and critically. Point out loopholes, hidden assumptions, and possible misconsiderations directly. I will be very happy to be convinced of my errors.

# Repository Guidelines
## Build, Lint, and Debug Commands
Always use `uv` for local commands; never invoke `pip`, `python`, `pytest`, or `modal` directly. Always use Modal for project execution, including snippets, tests, and experiments. Keep local work to editing and lightweight static inspection.

- `uv sync --project scripts/modal --locked`: create or update the persistent local Modal tooling environment.
- `uv run --project scripts/modal ruff check <changed-files>`: lint only changed code.
- `uv run --project scripts/modal modal run scripts/modal/train_modal.py --gpu-count <1-8> -- <slime-args>`: run synchronous training remotely.
- `uv run --project scripts/modal modal run scripts/modal/train_async_modal.py --gpu-count <1-8> -- <slime-args>`: run asynchronous training remotely.

The Qwen2.5-0.5B examples are prepared once and then run with:

- `uv run --project scripts/modal bash scripts/modal/examples/prepare-qwen2.5-0.5B.sh`
- `uv run --project scripts/modal bash scripts/modal/examples/run-qwen2.5-0.5B.sh`
- `uv run --project scripts/modal bash scripts/modal/examples/run-qwen2.5-0.5B-async.sh`

For debugging, prefer small Modal snippets that exercise one import, shape, or control-flow path.

## Modal Runtime
Modal is the runtime for all Python execution. Keep Modal entrypoints thin: they should configure images, secrets, volumes, and dispatch to importable Python modules. Place entrypoint scripts and their local tooling in `scripts/modal/`. Dataset and model files belong in the `code-maxrl-slime` Modal Volume; do not assume they exist locally.

### Modal Log Retrieval
For recent failures, recover logs via app id, then container id:

- `uv run --project scripts/modal modal app list --json`: find the latest matching `description` and copy its `app_id`.
- `uv run --project scripts/modal modal app logs <app-id> --show-container-id --tail 10`: recover the `container_id`.
- `uv run --project scripts/modal modal container logs <container-id> --all --timestamps`: inspect full logs. Use `--tail` instead of `--all` to limit output.

## Coding Style and Naming Conventions

Use full-path imports and minimal `__init__.py` files. Prefer explicit, importable functions and config objects over hidden defaults scattered across scripts.

Prefer explicit failure over permissive branching or broad `try/except`. When changing an API, make the clean breaking change instead of adding compatibility aliases or shims.

Before adding shared functionality, check for duplicate or reusable code in `slime/`, `slime_plugins/`, and `scripts/`.

## Testing Guidelines
This is a research repo, so snippet validation is the default. Do not run the full test suite unless explicitly requested. Use focused Modal snippets to validate imports, shapes, and control flow. If validation needs more than a short snippet, add or update a focused `test_<feature>.py` file under `tests/` and run only the relevant test node.
