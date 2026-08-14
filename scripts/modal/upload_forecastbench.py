from __future__ import annotations

from datetime import datetime
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = REPO_ROOT / "forecast_data" / "outputs"
REMOTE_OUTPUT_DIRECTORY = "/forecast_data/outputs"
VOLUME_NAME = "code-maxrl-slime"

ARTIFACT_PATTERNS = (
    "forecastbench_train_cutoff_{cutoff}.parquet",
    "forecastbench_eval_time_cutoff_{cutoff}.parquet",
    "forecastbench_eval_event_cutoff_{cutoff}.parquet",
    "forecastbench_analysis_cutoff_{cutoff}.json",
    "forecastbench_analysis_cutoff_{cutoff}.txt",
    "forecastbench_dist_cutoff_{cutoff}.png",
    "forecastbench_tokens_cutoff_{cutoff}.png",
)

app = modal.App("slime-upload-forecastbench")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.local_entrypoint()
def main(cutoff: str, output_dir: str = str(DEFAULT_OUTPUT_DIRECTORY)) -> None:
    cutoff_tag = _normalize_cutoff(cutoff)
    local_directory = Path(output_dir).resolve()
    artifacts = [local_directory / pattern.format(cutoff=cutoff_tag) for pattern in ARTIFACT_PATTERNS]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing ForecastBench artifacts:\n{formatted}")

    with volume.batch_upload(force=True) as upload:
        for path in artifacts:
            upload.put_file(str(path), f"{REMOTE_OUTPUT_DIRECTORY}/{path.name}")

    print(f"Uploaded {len(artifacts)} artifacts for cutoff {cutoff_tag} to Modal Volume {VOLUME_NAME}.")


def _normalize_cutoff(value: str) -> str:
    for date_format in ("%y%m%d", "%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, date_format).strftime("%y%m%d")
        except ValueError:
            continue
    raise ValueError(f"Expected cutoff as YYMMDD, YYYYMMDD, or YYYY-MM-DD; got {value!r}")
